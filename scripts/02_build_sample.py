from src.ingest.sample import build_monthly_sample
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/02_build_sample.log")

    build_monthly_sample(cfg)
    write_run_card("outputs/run_cards/02_build_sample.json", cfg)


if __name__ == "__main__":
    main()
