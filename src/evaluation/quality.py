import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def topic_size_table(topics: np.ndarray) -> pd.DataFrame:
    s = pd.Series(topics, name="topic_id")
    sizes = s.value_counts(dropna=False).reset_index()
    sizes.columns = ["topic_id", "n_docs"]
    sizes["share"] = sizes["n_docs"] / sizes["n_docs"].sum()
    return sizes.sort_values("n_docs", ascending=False)


def outlier_rate(topics: np.ndarray) -> float:
    topics = np.asarray(topics)
    return float((topics == -1).mean())


def within_topic_similarity(
    embeddings: np.ndarray,
    topics: np.ndarray,
    *,
    max_docs_per_topic: int = 50,
    max_topics: int = 200,
) -> pd.DataFrame:
    """
    Coherence proxy:
    - pick up to max_docs_per_topic docs from each topic
    - compute avg cosine similarity within that topic
    Higher means tighter clusters.
    """
    df = pd.DataFrame({"topic_id": topics})
    df["idx"] = np.arange(len(df))

    # Focus on largest topics (more stable estimate)
    top_topics = df["topic_id"].value_counts().head(max_topics).index.tolist()

    rows = []
    for t in top_topics:
        if t == -1:
            continue
        idxs = df.loc[df["topic_id"] == t, "idx"].head(max_docs_per_topic).to_numpy()
        if len(idxs) < 5:
            continue
        sims = cosine_similarity(embeddings[idxs], embeddings[idxs])
        # mean of upper triangle without diagonal
        n = sims.shape[0]
        mean_sim = (sims.sum() - n) / (n * (n - 1))
        rows.append(
            {"topic_id": int(t), "n_used": int(len(idxs)), "mean_cosine_sim": float(mean_sim)}
        )

    return pd.DataFrame(rows).sort_values("mean_cosine_sim", ascending=False)
