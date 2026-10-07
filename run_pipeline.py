"""Root entrypoint to execute the quantitative risk pipeline."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run import main

if __name__ == "__main__":
    main()
