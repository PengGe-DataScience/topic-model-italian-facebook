import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from bertopic import BERTopic

from src.modeling.embeddings import embed_texts, get_device

logger = logging.getLogger(__name__)


def _assigned_confidence(probs):
    """
    Handle version differences: sometimes probs is None / 1D / 2D.
    """
    if probs is None:
        return None
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs
    if probs.ndim == 2:
        return probs.max(axis=1)
    return None


def assign_topics_streaming(cfg: dict[str, Any]) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir = outputs_dir / "doc_topics"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg["paths"]["model_path"])
    topic_model = BERTopic.load(model_path.as_posix())

    emb_cfg = cfg["embedding"]
    device = get_device(emb_cfg["device"])

    # make sure year and month are in consistent format
    partition_schema = pa.schema(
        [
            ("year", pa.string()),
            ("month", pa.string()),
        ]
    )

    dataset = ds.dataset(
        processed_dir,
        format="parquet",
        partitioning=ds.partitioning(partition_schema, flavor="hive"),
    )

    fragments = list(dataset.get_fragments())
    logger.info("Assigning topics for %d parquet fragments", len(fragments))

    batch_size = int(cfg["streaming"]["parquet_batch_size"])

    for frag in fragments:
        scanner = ds.Scanner.from_fragment(
            frag,
            columns=["doc_id", "created_time", "year", "month", "text_clean"],
            batch_size=batch_size,
        )

        # Output partition folder from the first batch
        first_batch = None
        batches = []
        for b in scanner.to_batches():
            if first_batch is None:
                first_batch = b
            batches.append(b)

        if first_batch is None:
            continue

        year = str(first_batch.column(first_batch.schema.get_field_index("year"))[0].as_py())
        month = str(first_batch.column(first_batch.schema.get_field_index("month"))[0].as_py())

        out_partition = out_dir / f"year={year}" / f"month={month}"
        out_partition.mkdir(parents=True, exist_ok=True)
        out_file = out_partition / (Path(frag.path).stem + "_topics.parquet")

        # Collect rows in chunks so we don't keep embeddings around
        out_chunks = []
        for batch in batches:
            df = batch.to_pandas()
            texts = df["text_clean"].tolist()

            embeddings = embed_texts(
                texts,
                model_name=emb_cfg["model_name"],
                device=device,
                batch_size=int(emb_cfg["batch_size"]),
                normalize_embeddings=bool(emb_cfg["normalize_embeddings"]),
                max_seq_length=int(emb_cfg["max_seq_length"]),
            )

            topics, probs = topic_model.transform(texts, embeddings)
            conf = _assigned_confidence(probs)

            out_df = pd.DataFrame(
                {
                    "doc_id": df["doc_id"].astype(str),
                    "created_time": df["created_time"],
                    "year": df["year"].astype(str),
                    "month": df["month"].astype(str),
                    "topic_id": topics,
                }
            )
            if conf is not None:
                out_df["topic_confidence"] = conf

            out_chunks.append(out_df)

        final = pd.concat(out_chunks, ignore_index=True)
        final.to_parquet(out_file, index=False)
        logger.info("Wrote topic assignments: %s", out_file)

    logger.info("Done assigning topics.")
