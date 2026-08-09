"""Run all step verifiers in order. Exits non-zero on any failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Core data verifiers (always run).
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
    if rc != 0:
        return rc

    # Tag-specific O2 verifiers (politics only — finance has its own artefacts
    # in the parent onchain/ directory and is run separately).
    politics_dir = HERE / "onchain" / "politics"
    if politics_dir.exists():
        for ph in ("a", "b", "c", "d"):
            print(f"\n=== signal_lab.onchain.verify_o2 --phase {ph} --tag Politics ===")
            r = subprocess.run(
                [sys.executable, "-m", "signal_lab.onchain.verify_o2",
                 "--phase", ph, "--tag", "Politics",
                 "--out-dir", str(politics_dir)],
                capture_output=True, text=True, cwd=cwd,
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
