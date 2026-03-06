import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_monthly_sample(cfg: dict[str, Any]) -> None:
    raw_path = cfg["paths"]["raw_parquet"]
    sample_path = Path(cfg["paths"]["sample_path"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    text_col = cfg["data_schema"]["text_col"]
    ts_col = cfg["data_schema"]["timestamp_col"]
    acct_col = cfg["data_schema"]["account_col"]
    post_id_col = cfg["data_schema"].get("post_id_col")

    per_month = int(cfg["sampling"]["per_month"])
    seed = int(cfg["sampling"]["seed"])

    # Use a deterministic ordering key instead of random() for reproducibility
    con = duckdb.connect(cfg["paths"]["duckdb_path"])

    con.execute(
        f"""
        CREATE OR REPLACE VIEW base_for_sample AS
        SELECT
          {_sql_ident(text_col)} AS text,
          try_cast({_sql_ident(ts_col)} AS TIMESTAMP) AS created_time,
          {_sql_ident(acct_col)} AS account,
          {(_sql_ident(post_id_col) + " AS post_id") if post_id_col else "NULL AS post_id"},
          strftime(try_cast({_sql_ident(ts_col)} AS TIMESTAMP), '%Y') AS year,
          strftime(try_cast({_sql_ident(ts_col)} AS TIMESTAMP), '%m') AS month,
          -- stable doc_id
          md5(
            coalesce(cast({_sql_ident(acct_col)} AS VARCHAR), '') || '|' ||
            coalesce(cast({(_sql_ident(post_id_col) if post_id_col else "''")} AS VARCHAR), '') || '|' ||
            coalesce(cast({_sql_ident(ts_col)} AS VARCHAR), '')
          ) AS doc_id
        FROM read_parquet('{raw_path}')
        WHERE {_sql_ident(text_col)} IS NOT NULL
          AND length(trim({_sql_ident(text_col)})) > 0
          AND try_cast({_sql_ident(ts_col)} AS TIMESTAMP) IS NOT NULL;
        """
    )

    # Deterministic pseudo-random ordering by hashing doc_id + seed
    con.execute(
        f"""
        COPY (
          WITH ranked AS (
            SELECT
              *,
              row_number() OVER (
                PARTITION BY year, month
                ORDER BY hash(doc_id || '{seed}')  -- deterministic order
              ) AS rn
            FROM base_for_sample
          )
          SELECT
            doc_id, account, post_id, created_time, year, month, text
          FROM ranked
          WHERE rn <= {per_month}
        )
        TO '{sample_path.as_posix()}'
        (FORMAT PARQUET);
        """
    )

    logger.info("Wrote sample to %s", sample_path)
    con.close()
