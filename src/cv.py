"""Reusable evaluation runners: time-based CV, and the unseen-city stress test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from splits import make_folds, prepare_fold
from metrics import evaluate


def run_cv(make_model, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, tr_idx, te_idx in make_folds(raw):
        train, test, _, _ = prepare_fold(raw, tr_idx, te_idx)
        model = make_model().fit(train)
        rows.append({"fold": k, **evaluate(test["posted_rate"], model.predict(test), test["is_outlier"])})
    return pd.DataFrame(rows).set_index("fold")


def run_hard(make_model, raw: pd.DataFrame, seeds=(0, 1, 2, 3, 4), n_cities=8) -> pd.DataFrame:
    """Fold 3 with n_cities removed from training; score only on rows touching them."""
    _, tr_idx, te_idx = list(make_folds(raw))[-1]
    cities = sorted(set(raw.loc[tr_idx, "pickup"]) | set(raw.loc[tr_idx, "delivery"]))
    rows = []
    for s in seeds:
        held = set(np.random.default_rng(s).choice(cities, size=n_cities, replace=False))
        touches = lambda d: (d["pickup"].isin(held) | d["delivery"].isin(held)).to_numpy()
        tr, te = tr_idx[~touches(raw.loc[tr_idx])], te_idx[touches(raw.loc[te_idx])]
        train, test, _, _ = prepare_fold(raw, tr, te)    # cleaner refit without the held cities
        model = make_model().fit(train)
        rows.append({"seed": s, "test_rows": len(test),
                     **evaluate(test["posted_rate"], model.predict(test), test["is_outlier"])})
    return pd.DataFrame(rows).set_index("seed")