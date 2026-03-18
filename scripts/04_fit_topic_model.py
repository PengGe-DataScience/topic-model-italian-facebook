import logging
from pathlib import Path

import pandas as pd

from src.modeling.embeddings import embed_texts, get_device, load_embedding_model
from src.modeling.train_bertopic import fit_bertopic
from src.preprocess.text_cleaning import batch_clean_texts
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/04_fit_topic_model.log")

    sample_path = Path(cfg["paths"]["sample_path"])
    df = pd.read_parquet(sample_path)

    # Clean sample for embedding/modeling
    pp = cfg["preprocess"]
    docs_clean = batch_clean_texts(
        df["text"].tolist(),
        replace_urls=pp["replace_urls"],
        replace_mentions=pp["replace_mentions"],
        normalize_hashtags=pp["normalize_hashtags"],
        keep_emojis_as_text=pp["keep_emojis_as_text"],
        lowercase=pp["lowercase"],
    )

    # Embeddings (sample only)
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

    topic_model, topic_info = fit_bertopic(cfg, docs_clean, embeddings)

    # Save model + topics
    model_path = Path(cfg["paths"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(model_path.as_posix())

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    topic_info.to_parquet(outputs_dir / "topics.parquet", index=False)

    write_run_card(
        "outputs/run_cards/04_fit_topic_model.json",
        cfg,
        extra={
            "embedding_device": device,
            "n_sample_docs": len(df),
            "n_topics": int(topic_info.shape[0]),
        },
    )

    logger.info("Saved model to %s and topics to outputs/topics.parquet", model_path)


if __name__ == "__main__":
    main()
