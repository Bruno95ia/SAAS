"""CLI entry-point to start the SAAS live inference pipeline."""
from __future__ import annotations

import argparse
import logging

from saas.live_yolo import LiveYOLOService
from saas.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAAS live inference pipeline")
    parser.add_argument("--once", action="store_true", help="Process a single pass for debugging")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    logging.getLogger(__name__).info("Bootstrapping SAAS pipeline")
    args = parse_args()
    service = LiveYOLOService()

    if args.once:
        service.start()
        logging.getLogger(__name__).info("Running in single-pass mode for 5 seconds")
        try:
            import time

            time.sleep(5)
        finally:
            service.stop()
    else:
        service.run_forever()


if __name__ == "__main__":
    main()
