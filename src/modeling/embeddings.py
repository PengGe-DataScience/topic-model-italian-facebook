import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device(device_cfg: str) -> str:
    """
    device_cfg: "auto" | "cpu" | "cuda"
    """
    if device_cfg in {"cpu", "cuda"}:
        return device_cfg

    # auto:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    normalize_embeddings: bool,
    max_seq_length: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    logger.info("Embedding %d docs with %s on %s", len(texts), model_name, device)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    # Ensure float32 to reduce memory
    return emb.astype(np.float32, copy=False)
