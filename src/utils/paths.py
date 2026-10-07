"""Project path definitions."""
from pathlib import Path

# Project Root is two levels up from this file: src/utils/paths.py -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_MANUAL_DIR = DATA_RAW_DIR / "manual"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    for d in [
        CONFIGS_DIR,
        DATA_RAW_DIR,
        DATA_RAW_MANUAL_DIR,
        DATA_PROCESSED_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DOCS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
