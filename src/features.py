"""Feature groups, so each idea can be switched on and off in experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _base(d):
    return pd.DataFrame({
        "log_distance": np.log(d["distance"]),
        "is_reefer": (d["equipment"] == "Reefer").astype(int),
        "is_flatbed": (d["equipment"] == "Flatbed").astype(int),
    }, index=d.index)


def _weight(d):
    return d[["weight", "weight_capped", "weight_floored", "weight_missing", "weight_was_negative"]]


def _market(d):
    return d[["market_index", "market_index_missing"]]


def _market_smooth(d):
    """7-day centred rolling mean of the daily market_index: keeps the slow level,
    removes the weekly cycle. Uses features only, never the target."""
    daily = d.groupby("date")["market_index"].mean().sort_index()
    smooth = daily.rolling(7, center=True, min_periods=4).mean()
    return pd.DataFrame({"mi_smooth": d["date"].map(smooth)}, index=d.index)


def _quote(d):
    return d[["quote_signal"]]


def _dow(d):
    return pd.DataFrame({"dow": d["date"].dt.dayofweek}, index=d.index)


def _dow_dom(d):
    return pd.DataFrame({"dow": d["date"].dt.dayofweek, "dom": d["date"].dt.day}, index=d.index)


def _month(d):            # RISKY: the model has never seen Nov or Dec
    return pd.DataFrame({"month": d["date"].dt.month}, index=d.index)


def _coords(d):
    return d[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]]

def _qs_month(d):
    """Month-level mean of quote_signal within this dataframe, plus the row's deviation
    from it. Features only, no target."""
    ym = d["date"].dt.to_period("M")
    m = d.groupby(ym)["quote_signal"].transform("mean")
    return pd.DataFrame({"qs_month_mean": m, "qs_dev": d["quote_signal"] - m}, index=d.index)

def _qs_dev(d):
    return _qs_month(d)[["qs_dev"]]
GROUPS = {
    "base": _base,
    "weight": _weight,
    "market_index": _market,
    "market_smooth": _market_smooth,
    "quote_signal": _quote,
    "dow": _dow,
    "dow_dom": _dow_dom,
    "month": _month,
    "coords": _coords,
    "qs_month": _qs_month,
    "qs_dev": _qs_dev,
}


def build_X(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return pd.concat([GROUPS[g](df) for g in groups], axis=1)