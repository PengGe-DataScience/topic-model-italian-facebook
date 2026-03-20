import logging
from pathlib import Path
from typing import Any

import duckdb
from bertopic import BERTopic

logger = logging.getLogger(__name__)


def make_reports(cfg: dict[str, Any]) -> None:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    model_path = Path(cfg["paths"]["model_path"])
    topic_model = BERTopic.load(model_path.as_posix())

    # 1) BERTopic interactive visuals (run on the trained model)
    # Note: these are built on the training corpus / topic embeddings
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(reports_dir / "topics_overview.html")

    fig_barchart = topic_model.visualize_barchart(top_n_topics=30)
    fig_barchart.write_html(reports_dir / "topics_barchart_top30.html")

    # 2) Trends over time using DuckDB aggregations on doc_topics parquet
    con = duckdb.connect(cfg["paths"]["duckdb_path"])
    doc_topics_glob = (outputs_dir / "doc_topics" / "**" / "*.parquet").as_posix()

    # Make monthly counts per topic
    trends = con.execute(
        f"""
        SELECT
          date_trunc('month', try_cast(created_time AS TIMESTAMP)) AS month,
          topic_id,
          COUNT(*) AS n_posts
        FROM read_parquet('{doc_topics_glob}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()

    trends_path = outputs_dir / "topic_trends.parquet"
    trends.to_parquet(trends_path, index=False)

    # Simple Plotly line chart (top 20 topics by volume)
    import plotly.express as px

    top_topics = (
        trends.groupby("topic_id")["n_posts"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    trends_top = trends[trends["topic_id"].isin(top_topics)].copy()

    fig_trends = px.line(
        trends_top, x="month", y="n_posts", color="topic_id", title="Topic trends (Top 20 topics)"
    )
    fig_trends.write_html(reports_dir / "topic_trends.html")

    logger.info("Wrote HTML reports and outputs/topic_trends.parquet")
    con.close()
