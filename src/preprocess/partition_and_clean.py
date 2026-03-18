import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.preprocess.text_cleaning import batch_clean_texts

logger = logging.getLogger(__name__)


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def partition_raw_by_month(cfg: dict[str, Any]) -> None:
    """
    Creates data/interim/raw_partitioned/year=YYYY/month=MM/*.parquet
    so later steps can iterate month-by-month.
    """
    raw_path = cfg["paths"]["raw_parquet"]
    out_dir = Path(cfg["paths"]["interim_partitioned_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    text_col = cfg["data_schema"]["text_col"]
    ts_col = cfg["data_schema"]["timestamp_col"]
    acct_col = cfg["data_schema"]["account_col"]
    post_id_col = cfg["data_schema"].get("post_id_col")

    con = duckdb.connect(cfg["paths"]["duckdb_path"])

    # COPY PARTITION_BY writes hive-style directories year=YYYY/month=MM/
    con.execute(
        f"""
        COPY (
          SELECT
            -- stable doc_id
            md5(
              coalesce(cast({_sql_ident(acct_col)} AS VARCHAR), '') || '|' ||
              coalesce(cast({(_sql_ident(post_id_col) if post_id_col else "''")} AS VARCHAR), '') || '|' ||
              coalesce(cast({_sql_ident(ts_col)} AS VARCHAR), '')
            ) AS doc_id,
            {_sql_ident(acct_col)} AS account,
            {(_sql_ident(post_id_col) + " AS post_id") if post_id_col else "NULL AS post_id"},
            try_cast({_sql_ident(ts_col)} AS TIMESTAMP) AS created_time,
            strftime(try_cast({_sql_ident(ts_col)} AS TIMESTAMP), '%Y') AS year,
            strftime(try_cast({_sql_ident(ts_col)} AS TIMESTAMP), '%m') AS month,
            {_sql_ident(text_col)} AS text
          FROM read_parquet('{raw_path}')
          WHERE {_sql_ident(text_col)} IS NOT NULL
            AND length(trim({_sql_ident(text_col)})) > 0
            AND try_cast({_sql_ident(ts_col)} AS TIMESTAMP) IS NOT NULL
        )
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE TRUE);
        """
    )

    logger.info("Partitioned raw parquet into %s", out_dir)
    con.close()


def clean_partitioned_dataset(cfg: dict[str, Any]) -> None:
    """
    Reads the partitioned raw dataset and writes cleaned dataset:
      data/processed/posts_clean/year=YYYY/month=MM/*.parquet
    """
    in_dir = Path(cfg["paths"]["interim_partitioned_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pp = cfg["preprocess"]
    batch_size = int(cfg["streaming"]["parquet_batch_size"])

    dataset = ds.dataset(in_dir, format="parquet", partitioning="hive")

    fragments = list(dataset.get_fragments())
    logger.info("Found %d parquet fragments to clean", len(fragments))

    for frag in fragments:
        m = re.search(r"year=(\d{4}).*month=(\d{2})", frag.path.replace("\\", "/"))
        if not m:
            logger.warning("Could not parse year/month from fragment path: %s", frag.path)
            continue
        year, month = m.group(1), m.group(2)

        scanner = ds.Scanner.from_fragment(
            frag,
            columns=["doc_id", "account", "post_id", "created_time", "text"],
            batch_size=batch_size,
        )

        out_partition_dir = out_dir / f"year={year}" / f"month={month}"
        out_partition_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_partition_dir / (Path(frag.path).stem + "_clean.parquet")

        writer = None
        try:
            for batch in scanner.to_batches():
                table = pa.Table.from_batches([batch])
                df = table.to_pandas()

                df["year"] = year
                df["month"] = month

                df["text_clean"] = batch_clean_texts(
                    df["text"].tolist(),
                    replace_urls=pp["replace_urls"],
                    replace_mentions=pp["replace_mentions"],
                    normalize_hashtags=pp["normalize_hashtags"],
                    keep_emojis_as_text=pp["keep_emojis_as_text"],
                    lowercase=pp["lowercase"],
                )

                # Drop short docs after cleaning
                min_chars = int(pp["min_chars"])
                df = df[df["text_clean"].str.len() >= min_chars].copy()

                schema = pa.schema(
                    [
                        ("doc_id", pa.string()),
                        ("account", pa.string()),
                        ("post_id", pa.string()),
                        ("created_time", pa.timestamp("ns")),
                        ("text_clean", pa.string()),
                    ]
                )

                out_table = pa.Table.from_pandas(
                    df[["doc_id", "account", "post_id", "created_time", "text_clean"]],
                    schema=schema,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file.as_posix(), out_table.schema, compression="zstd"
                    )
                writer.write_table(out_table)

        finally:
            if writer is not None:
                writer.close()

        logger.info("Wrote cleaned partition file: %s", out_file)
