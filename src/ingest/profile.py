import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def _sql_ident(name: str) -> str:
    # Quote identifiers safely for DuckDB SQL
    return '"' + name.replace('"', '""') + '"'


def profile_parquet(cfg: dict[str, Any]) -> None:
    raw_path = cfg["paths"]["raw_parquet"]
    db_path = cfg["paths"]["duckdb_path"]
    text_col = cfg["data_schema"]["text_col"]
    ts_col = cfg["data_schema"]["timestamp_col"]
    acct_col = cfg["data_schema"]["account_col"]
    post_id_col = cfg["data_schema"].get("post_id_col")

    reports_dir = Path(cfg["paths"]["reports_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)

    # A "virtual table" view over parquet (no full import)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_posts AS
        SELECT
            {_sql_ident(text_col)} AS text,
            {_sql_ident(ts_col)}   AS created_time,
            {_sql_ident(acct_col)} AS account,
            {(_sql_ident(post_id_col) + " AS post_id") if post_id_col else "NULL AS post_id"}
        FROM read_parquet('{raw_path}');
        """
    )

    # Basic counts
    total = con.execute("SELECT COUNT(*) AS n FROM raw_posts").fetchdf().iloc[0]["n"]

    null_text = (
        con.execute(
            "SELECT AVG(CASE WHEN text IS NULL OR length(trim(text)) = 0 THEN 1 ELSE 0 END) AS p FROM raw_posts"
        )
        .fetchdf()
        .iloc[0]["p"]
    )

    # Date coverage
    coverage = con.execute(
        """
        SELECT
          MIN(try_cast(created_time AS TIMESTAMP)) AS min_time,
          MAX(try_cast(created_time AS TIMESTAMP)) AS max_time
        FROM raw_posts
        """
    ).fetchdf()

    # Text length distribution
    lengths = con.execute(
        """
        SELECT
          AVG(length(text)) AS avg_len,
          approx_quantile(length(text), 0.5) AS p50_len,
          approx_quantile(length(text), 0.9) AS p90_len,
          approx_quantile(length(text), 0.99) AS p99_len
        FROM raw_posts
        WHERE text IS NOT NULL
        """
    ).fetchdf()

    # Duplicates if you have post_id
    dup_df = None
    if post_id_col:
        dup_df = con.execute(
            """
            SELECT
              COUNT(*) AS n,
              COUNT(DISTINCT post_id) AS n_distinct,
              (COUNT(*) - COUNT(DISTINCT post_id)) AS n_duplicates
            FROM raw_posts
            WHERE post_id IS NOT NULL
            """
        ).fetchdf()

    # Volume by month
    monthly = con.execute(
        """
        SELECT
          date_trunc('month', try_cast(created_time AS TIMESTAMP)) AS month,
          COUNT(*) AS n_posts
        FROM raw_posts
        WHERE try_cast(created_time AS TIMESTAMP) IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    # Top accounts
    top_accounts = con.execute(
        """
        SELECT account, COUNT(*) AS n_posts
        FROM raw_posts
        WHERE account IS NOT NULL
        GROUP BY 1
        ORDER BY n_posts DESC
        LIMIT 25
        """
    ).fetchdf()

    # Write profiling tables to reuse later
    (outputs_dir / "profile").mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(outputs_dir / "profile" / "monthly_counts.parquet", index=False)
    top_accounts.to_parquet(outputs_dir / "profile" / "top_accounts.parquet", index=False)

    # Markdown report
    report_lines = []
    report_lines.append("# Data profile\n")
    report_lines.append(f"- Total rows: **{total:,}**\n")
    report_lines.append(f"- Null/empty text ratio: **{null_text:.3f}**\n")
    report_lines.append(f"- Min date: **{coverage.iloc[0]['min_time']}**\n")
    report_lines.append(f"- Max date: **{coverage.iloc[0]['max_time']}**\n")
    report_lines.append("\n## Text length\n")
    report_lines.append(lengths.to_markdown(index=False))
    report_lines.append("\n\n## Duplicates (by post_id)\n")
    if dup_df is not None:
        report_lines.append(dup_df.to_markdown(index=False))
    else:
        report_lines.append("_post_id_col not provided; skipping duplicate count._")

    report_lines.append("\n\n## Monthly volume (first/last 12)\n")
    report_lines.append(pd.concat([monthly.head(12), monthly.tail(12)]).to_markdown(index=False))
    report_lines.append("\n\n## Top accounts\n")
    report_lines.append(top_accounts.to_markdown(index=False))

    (reports_dir / "data_profile.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Wrote reports/data_profile.md and outputs/profile/*.parquet")
    con.close()
