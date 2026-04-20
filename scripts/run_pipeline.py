"""
End-to-end pipeline runner.
Usage:
  python scripts/run_pipeline.py --stage all
  python scripts/run_pipeline.py --stage ingest
  python scripts/run_pipeline.py --stage preprocess
  python scripts/run_pipeline.py --stage train
  python scripts/run_pipeline.py --stage serve
"""

import argparse
import sys
from loguru import logger


def run_ingest():
    logger.info("=== Stage 1: Data Ingestion ===")
    from src.data.ingest import main
    main()


def run_preprocess():
    logger.info("=== Stage 2: Preprocessing ===")
    from src.data.preprocess import run_preprocessing
    run_preprocessing()


def run_train():
    logger.info("=== Stage 3: Training & Model Selection ===")
    from src.training.train import main
    main()


# def run_serve():
#     logger.info("=== Stage 4: Starting API server ===")
#     import uvicorn
#     uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

def run_serve():
    logger.info("=== Stage 4: Starting API server ===")
    import uvicorn, yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    uvicorn.run(
        "src.api.app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,  # disable reload — avoids duplicate startup
    )


STAGES = {
    "ingest": run_ingest,
    "preprocess": run_preprocess,
    "train": run_train,
    "serve": run_serve,
}


def main():
    parser = argparse.ArgumentParser(description="MLOps Sentiment Pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Which stage to run",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name, fn in STAGES.items():
            if name == "serve":
                continue
            try:
                fn()
                logger.success(f"Stage '{name}' completed.")
            except Exception as e:
                logger.error(f"Stage '{name}' failed: {e}")
                sys.exit(1)
        logger.success("Pipeline complete! Start the API with: python scripts/run_pipeline.py --stage serve")
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
