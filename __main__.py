"""`python -m PALM <config.yaml>` -- delegates to the DataSAIL pipeline CLI."""

import sys

from PALM.benchmarks.datasail_pipeline.__main__ import main

if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("Usage: python -m PALM <config.yaml>\n"
              "Runs the DataSAIL-style splitting pipeline "
              "(PALM.benchmarks.datasail_pipeline).")
        sys.exit(0)
    main()
