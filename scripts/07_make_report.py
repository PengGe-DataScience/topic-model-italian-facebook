from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card
from src.visualization.reporting import make_reports


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/07_make_report.log")

    make_reports(cfg)
    write_run_card("outputs/run_cards/07_make_report.json", cfg)


if __name__ == "__main__":
    main()
