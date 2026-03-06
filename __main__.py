"""Entry point: python -m PALM config.yaml"""

import sys
from .config import load_config
from .pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m PALM <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
