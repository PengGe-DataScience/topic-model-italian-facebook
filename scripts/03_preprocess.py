from src.preprocess.partition_and_clean import clean_partitioned_dataset, partition_raw_by_month
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.run_card import write_run_card


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    setup_logging("outputs/logs/03_preprocess.log")

    partition_raw_by_month(cfg)
    clean_partitioned_dataset(cfg)

    write_run_card("outputs/run_cards/03_preprocess.json", cfg)


if __name__ == "__main__":
    main()
