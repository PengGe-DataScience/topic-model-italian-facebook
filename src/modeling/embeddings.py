import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device(device_cfg: str) -> str:
    if device_cfg in {"cpu", "cuda"}:
        return device_cfg

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_embedding_model(
    *,
    model_name: str,
    device: str,
    max_seq_length: int,
) -> SentenceTransformer:
    """
    Load the sentence-transformer once and reuse it across batches.
    """
    logger.info("Loading embedding model %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def embed_texts(
    texts: list[str],
    *,
    model: SentenceTransformer,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    """
    Encode a batch of texts using an already-loaded model.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.astype(np.float32, copy=False)
