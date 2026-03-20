import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic

from src.evaluation.quality import outlier_rate, topic_size_table, within_topic_similarity
from src.modeling.embeddings import embed_texts, get_device, load_embedding_model
from src.preprocess.text_cleaning import batch_clean_texts
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/06_evaluate_and_refine.log")

    # Evaluate on the sample again
    sample_path = Path(cfg["paths"]["sample_path"])
    df = pd.read_parquet(sample_path)

    pp = cfg["preprocess"]
    docs_clean = batch_clean_texts(
        df["text"].tolist(),
        replace_urls=pp["replace_urls"],
        replace_mentions=pp["replace_mentions"],
        normalize_hashtags=pp["normalize_hashtags"],
        keep_emojis_as_text=pp["keep_emojis_as_text"],
        lowercase=pp["lowercase"],
    )

    model_path = Path(cfg["paths"]["model_path"])
    topic_model = BERTopic.load(model_path.as_posix())

    emb_cfg = cfg["embedding"]
    device = get_device(emb_cfg["device"])

    embedding_model = load_embedding_model(
        model_name=emb_cfg["model_name"],
        device=device,
        max_seq_length=int(emb_cfg["max_seq_length"]),
    )

    embeddings = embed_texts(
        docs_clean,
        model=embedding_model,
        batch_size=int(emb_cfg["batch_size"]),
        normalize_embeddings=bool(emb_cfg["normalize_embeddings"]),
    )

    topics, probs = topic_model.transform(docs_clean, embeddings)

    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = topic_size_table(np.asarray(topics))
    sizes.to_parquet(out_dir / "topic_sizes_sample.parquet", index=False)

    o_rate = outlier_rate(np.asarray(topics))
    logger.info("Outlier rate on sample: %.3f", o_rate)

    sim_df = within_topic_similarity(
        embeddings,
        np.asarray(topics),
        max_docs_per_topic=50,
        max_topics=200,
    )
    sim_df.to_parquet(out_dir / "topic_similarity_proxy.parquet", index=False)

    summary = {
        "outlier_rate_sample": o_rate,
        "n_topics_in_model": int(topic_model.get_topic_info().shape[0]),
        "embedding_device": device,
    }
    write_run_card("outputs/run_cards/06_evaluate_and_refine.json", cfg, extra=summary)
    logger.info("Wrote evaluation outputs to outputs/*.parquet")


if __name__ == "__main__":
    main()
