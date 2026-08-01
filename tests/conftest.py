import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "wallet_selection"
sys.path.insert(0, str(NOTEBOOK_DIR))
