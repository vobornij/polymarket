"""Run all step verifiers in order. Exits non-zero on any failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VERIFIERS = [
    "signal_lab.weather_fv.verify_w0_1",
    "signal_lab.weather_fv.verify_w0_2",
    "signal_lab.weather_fv.verify_w0_3",
    "signal_lab.weather_fv.verify_w1",
    "signal_lab.onchain.verify_o1",
]


def main() -> int:
    rc = 0
    cwd = HERE.parent  # notebooks/wallet_selection
    for v in VERIFIERS:
        print(f"\n=== {v} ===")
        r = subprocess.run(
            [sys.executable, "-m", v], capture_output=True, text=True, cwd=cwd
        )
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            rc = r.returncode
            break
    if rc == 0:
        print("\nALL OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
