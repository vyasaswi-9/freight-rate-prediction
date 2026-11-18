"""Metrics reported on all rows and on rows without flagged outliers."""
from __future__ import annotations

import numpy as np


def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape(y, p): return float(np.mean(np.abs(y - p) / y) * 100)


def evaluate(y_true, y_pred, is_outlier=None) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE%": mape(y, p)}
    if is_outlier is not None:
        keep = ~np.asarray(is_outlier, dtype=bool)
        out.update({"MAE_clean": mae(y[keep], p[keep]),
                    "RMSE_clean": rmse(y[keep], p[keep]),
                    "MAPE%_clean": mape(y[keep], p[keep])})
    return out