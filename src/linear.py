"""Log-linear model on log(rate per mile) with polynomial distance terms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "posted_rate"


class LogLinearModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _X(self, d: pd.DataFrame) -> pd.DataFrame:
        ld = np.log(d["distance"])
        w = d["weight"] / 10_000
        reefer = (d["equipment"] == "Reefer").astype(int)
        flatbed = (d["equipment"] == "Flatbed").astype(int)
        X = pd.DataFrame({
            "ld": ld, "ld2": ld ** 2, "ld3": ld ** 3,
            "reefer": reefer, "flatbed": flatbed,
            "reefer_ld": reefer * ld, "flatbed_ld": flatbed * ld,
            "w": w, "w2": w ** 2,
            "capped": d["weight_capped"], "floored": d["weight_floored"],
            "w_missing": d["weight_missing"], "w_neg": d["weight_was_negative"],
        }, index=d.index)
        dow = d["date"].dt.dayofweek
        for k in range(1, 7):
            X[f"dow{k}"] = (dow == k).astype(int)
        return X

    def fit(self, train: pd.DataFrame):
        d = train[~train["is_outlier"]]
        y = np.log(d[TARGET] / d["distance"])
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(self._X(d), y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(df["distance"].to_numpy() * np.exp(self.model_.predict(self._X(df))),
                         index=df.index)