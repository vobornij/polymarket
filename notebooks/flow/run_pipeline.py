#!/usr/bin/env python3
"""
Run all pipeline stages consecutively.

Usage:
    python flow/run_pipeline.py [--stages 0 1 2 3] [--timeout 600]

Each stage notebook is executed in-place (outputs are written back).
Progress and timing are logged to stdout and to run_pipeline.log in the
same directory as this script.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
VENV_JUPYTER = SCRIPT_DIR.parent.parent / ".venv" / "bin" / "jupyter"

STAGES = [
    {
        "id": 0,
        "name": "Polygon trade analysis (initial filtering)",
        "notebook": SCRIPT_DIR / "stage0_initial_filtering.ipynb",
    },
    {
        "id": 1,
        "name": "Profitable wallet analysis (volatility selector)",
        "notebook": SCRIPT_DIR / "stage1_profitable_wallet_analysis.ipynb",
    },
    {
        "id": 2,
        "name": "Wallet signal builder & calibration",
        "notebook": SCRIPT_DIR / "stage2_wallet_signal.ipynb",
    },
    {
        "id": 3,
        "name": "Backtest strategy sweep",
        "notebook": SCRIPT_DIR / "stage3_backtest.ipynb",
    },
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FILE = SCRIPT_DIR / "run_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _find_jupyter() -> str:
    """Return the path to the jupyter executable."""
    if VENV_JUPYTER.exists():
        return str(VENV_JUPYTER)
    # Fallback: use whatever is on PATH
    result = subprocess.run(["which", "jupyter"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    raise RuntimeError(
        "jupyter not found. Run this script from within the project virtualenv "
        "or install jupyter: poetry run pip install jupyter"
    )


def run_stage(stage: dict, timeout: int) -> bool:
    """
    Execute a single notebook stage in-place.

    Returns True on success, False on failure.
    """
    notebook = stage["notebook"]
    name = f"stage{stage['id']} – {stage['name']}"

    if not notebook.exists():
        log.error("  Notebook not found: %s", notebook)
        return False

    jupyter = _find_jupyter()
    cmd = [
        jupyter, "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]

    log.info("  Running: %s", " ".join(cmd))
    t0 = time.monotonic()

    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if proc.returncode == 0:
        log.info("  Completed in %s", _fmt_duration(elapsed))
        return True
    else:
        log.error("  FAILED after %s (exit code %d)", _fmt_duration(elapsed), proc.returncode)
        # nbconvert writes the Python traceback to stderr
        for line in proc.stderr.splitlines()[-30:]:
            log.error("    %s", line)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[s["id"] for s in STAGES],
        metavar="N",
        help="Stage IDs to run (default: all). E.g. --stages 2 3",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        metavar="SEC",
        help="Per-cell execution timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    selected = [s for s in STAGES if s["id"] in args.stages]
    if not selected:
        log.error("No matching stages found for --stages %s", args.stages)
        return 1

    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log.info("=" * 70)
    log.info("Pipeline run  %s", run_id)
    log.info("Stages:       %s", [s["id"] for s in selected])
    log.info("Timeout:      %ds per cell", args.timeout)
    log.info("Log file:     %s", LOG_FILE)
    log.info("=" * 70)

    pipeline_start = time.monotonic()
    failed = []

    for stage in selected:
        log.info("")
        log.info("─" * 70)
        log.info("Stage %d / %d  │  %s", stage["id"], STAGES[-1]["id"], stage["name"])
        log.info("─" * 70)

        ok = run_stage(stage, timeout=args.timeout)
        if not ok:
            failed.append(stage["id"])
            log.error("Aborting pipeline after stage %d failure.", stage["id"])
            break

    total = time.monotonic() - pipeline_start
    log.info("")
    log.info("=" * 70)
    if failed:
        log.error("Pipeline FAILED at stage %s  (total time: %s)", failed[0], _fmt_duration(total))
    else:
        log.info("Pipeline COMPLETE  (total time: %s)", _fmt_duration(total))
    log.info("=" * 70)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
