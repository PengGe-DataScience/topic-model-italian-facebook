import logging
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)


def _safe_assigned_probability(probs) -> pd.Series:
    """
    Convert all cases to a 1D "confidence" series when possible.
    """
    if probs is None:
        return pd.Series([None])

    probs = np.asarray(probs)
    if probs.ndim == 1:
        return pd.Series(probs)
    if probs.ndim == 2:
        return pd.Series(probs.max(axis=1))
    return pd.Series([None])


def fit_bertopic(
    cfg: dict[str, Any],
    docs_for_model: list[str],
    embeddings: np.ndarray,
) -> tuple[BERTopic, pd.DataFrame]:
    bt = cfg["bertopic"]

    vectorizer_cfg = bt["vectorizer"]
    vectorizer_model = CountVectorizer(
        ngram_range=tuple(vectorizer_cfg["ngram_range"]),
        min_df=int(vectorizer_cfg["min_df"]),
        max_df=float(vectorizer_cfg["max_df"]),
        max_features=int(vectorizer_cfg["max_features"]),
    )

    umap_cfg = bt["umap"]
    umap_model = UMAP(
        n_neighbors=int(umap_cfg["n_neighbors"]),
        n_components=int(umap_cfg["n_components"]),
        min_dist=float(umap_cfg["min_dist"]),
        metric=str(umap_cfg["metric"]),
        random_state=int(umap_cfg["random_state"]),
    )

    hdb_cfg = bt["hdbscan"]
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdb_cfg["min_cluster_size"]),
        min_samples=int(hdb_cfg["min_samples"]),
        metric=str(hdb_cfg["metric"]),
        cluster_selection_method=str(hdb_cfg["cluster_selection_method"]),
        prediction_data=bool(hdb_cfg["prediction_data"]),
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=bool(bt["calculate_probabilities"]),
        low_memory=bool(bt["low_memory"]),
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs_for_model, embeddings)

    info = topic_model.get_topic_info()
    logger.info("Trained model: %d topics (including -1)", info.shape[0])

    return topic_model, info
