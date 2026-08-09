"""O2 — simple direct price / lead / market rules evaluated against roi_res.

Each rule is a callable ``rule(frame) -> pd.Series[bool]`` applied to a
candidate frame. The boolean mask becomes a "fake composite score" and
its Spearman IC vs ``roi_res`` is computed on each split.

The frame must have:
  - ``price``  (float in [0, 1])
  - ``lead_h`` (float hours until market close; negative is post-close)
  - ``roi_res`` (price-residualized ROI; the existing ``roi_res`` column)
  - ``condition_id`` (string) — for the one-off-market rule

See also: ``o2_runner.py`` (the dispatcher) and the per-step verifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL_RULES: dict[str, callable] = {}


def _register(name: str):
    def deco(fn):
        ALL_RULES[name] = fn
        return fn
    return deco


@_register("price_lt_0p5_lead_gt_24h")
def price_lt_0p5_lead_gt_24h(frame: pd.DataFrame) -> pd.Series:
    return (frame["price"] < 0.5) & (frame["lead_h"] > 24)


@_register("price_lt_0p5_lead_gt_72h")
def price_lt_0p5_lead_gt_72h(frame: pd.DataFrame) -> pd.Series:
    return (frame["price"] < 0.5) & (frame["lead_h"] > 72)


@_register("price_lt_0p1")
def price_lt_0p1(frame: pd.DataFrame) -> pd.Series:
    return frame["price"] < 0.1


@_register("price_gt_0p9")
def price_gt_0p9(frame: pd.DataFrame) -> pd.Series:
    return frame["price"] > 0.9


@_register("price_mid_lead_gt_24h")
def price_mid_lead_gt_24h(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["price"] >= 0.3)
        & (frame["price"] <= 0.7)
        & (frame["lead_h"] > 24)
    )


@_register("recurring_market")
def recurring_market(frame: pd.DataFrame) -> pd.Series:
    """Politics-only: trade only on markets with >=3 distinct trading days.

    Requires ``n_distinct_days_per_market`` to be pre-computed in the frame
    by the runner; absent that, fall back to a 0/1 constant.
    """
    if "n_distinct_days_per_market" in frame.columns:
        days = frame["condition_id"].map(
            frame.drop_duplicates("condition_id").set_index("condition_id")["n_distinct_days_per_market"]
        )
        return days.fillna(0).astype(int) >= 3
    return pd.Series(False, index=frame.index)


POLITICS_ONLY = {"recurring_market"}


def rule_names_for_tag(tag: str) -> list[str]:
    if tag.lower() == "politics":
        return list(ALL_RULES.keys())
    return [r for r in ALL_RULES if r not in POLITICS_ONLY]


def evaluate_rule(
    frame: pd.DataFrame, rule_name: str
) -> dict:
    """Compute IC vs roi_res on the boolean mask (treated as 0/1 score)."""
    from scipy.stats import spearmanr

    rule = ALL_RULES[rule_name]
    mask = rule(frame).fillna(False).astype(int)
    if mask.sum() < 30:
        return {"rule": rule_name, "n_fires": int(mask.sum()),
                "ic": float("nan"), "mean_fired_roi_res": float("nan"),
                "mean_unfired_roi_res": float("nan")}
    y = frame["roi_res"].astype(float)
    if y.nunique() < 2:
        return {"rule": rule_name, "n_fires": int(mask.sum()),
                "ic": float("nan"), "mean_fired_roi_res": float("nan"),
                "mean_unfired_roi_res": float("nan")}
    fired = mask.astype(bool)
    ic, _ = spearmanr(mask, y, nan_policy="omit")
    return {
        "rule": rule_name,
        "n_total": int(len(frame)),
        "n_fires": int(fired.sum()),
        "fire_rate": float(fired.mean()),
        "ic": float(ic) if np.isfinite(ic) else float("nan"),
        "mean_fired_roi_res": float(y[fired].mean()) if fired.any() else float("nan"),
        "mean_unfired_roi_res": float(y[~fired].mean()) if (~fired).any() else float("nan"),
    }
