"""One-command, end-to-end FlowCast run (PRD §3.3 reproducibility):

    python -m src.pipeline --data-dir data/raw

Stages: M1 ingest -> M2 clean/merge -> M3 features -> M4 EDA -> M5 classical ML -> M6 GRU/LSTM
-> live forecast for the next 4 windows. The dashboard (M7) only reads what this writes.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.clean import run_cleaning
from src.eda import run_eda
from src.features import build_all_horizons
from src.utils import PROJECT_ROOT, ensure_dirs, get_logger, load_config, set_seed


def main():
    """Parse arguments and run every module in order."""
    parser = argparse.ArgumentParser(description="Run the FlowCast pipeline end to end.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"),
                        help="folder with traffic_sensor_log.csv, weather_observations.csv, calendar_events.csv")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode: 8 segments, tiny grids, 2 DL epochs")
    parser.add_argument("--skip-dl", action="store_true", help="skip the GRU/LSTM stage")
    arguments = parser.parse_args()

    logger = get_logger()
    config = load_config()
    set_seed(config["seed"])
    ensure_dirs(config)
    data_dir = Path(arguments.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_label = f"raw files in {data_dir.name}/"
    started = time.time()

    logger.info(f"=== M1+M2 ingest & clean ({data_label})")
    clean, dq_log = run_cleaning(data_dir, config)
    logger.info("=== M3 features")
    tables = build_all_horizons(clean, config)
    logger.info("=== M4 EDA + data-quality report")
    run_eda(clean, tables[1], dq_log, config, data_label)

    scoreboard_path = config["paths"]["reports_dir"] / "scoreboard.csv"
    if scoreboard_path.exists():
        scoreboard_path.unlink()                      # fresh scoreboard for every full run

    logger.info("=== M5 classical ML")
    from src.ml_models import run_classical

    run_classical(tables, config, fast=arguments.fast)

    if not arguments.skip_dl:
        logger.info("=== M6 deep learning")
        from src.dl_model import run_deep_learning

        run_deep_learning(clean, tables[1], config, fast=arguments.fast)

    logger.info("=== Live forecast (next 4 windows) with persisted models")
    from src.predict import predict_next_windows

    live = predict_next_windows(clean, config)
    live.to_parquet(config["paths"]["processed_dir"] / "live_predictions.parquet", index=False)

    run_info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "data_label": data_label,
        "fast_mode": arguments.fast,
        "deep_learning": not arguments.skip_dl,
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    with open(config["paths"]["reports_dir"] / "run_info.json", "w", encoding="utf-8") as info_file:
        json.dump(run_info, info_file, indent=2)
    logger.info(f"=== Done in {run_info['runtime_minutes']} min. Start the dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
