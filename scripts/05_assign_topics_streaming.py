from src.modeling.assign_topics import assign_topics_streaming
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/05_assign_topics_streaming.log")

    assign_topics_streaming(cfg)
    write_run_card("outputs/run_cards/05_assign_topics_streaming.json", cfg)


if __name__ == "__main__":
    main()
