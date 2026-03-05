from src.ingest.profile import profile_parquet
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/01_profile_sql.log")

    profile_parquet(cfg)
    write_run_card("outputs/run_cards/01_profile_sql.json", cfg)


if __name__ == "__main__":
    main()
