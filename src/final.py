"""Train the final model on all labelled data and write both submission files.

Run from the project root:  python src/final.py

Writes:
  validation_predictions.csv          load_id,predicted_rate for all 12,000 validation loads
  data/december_chart_inputs.csv      same file with its predicted_rate column filled in
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from clean import Cleaner, fit_rate_baseline, outlier_mask
from gbm import GBMModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FEATURES = ["base", "weight", "dow", "coords"]          # feature set S (see 05_features.ipynb)
PARAMS = dict(max_leaf_nodes=15, min_samples_leaf=100)  # "GBM small leaves" (see 06_models.ipynb)


def city_coords(*frames: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city, taken from the input files (coordinates are fixed per city).
    Needed because the December file has city names but no coordinates."""
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            cols = [side, f"{side}_lat", f"{side}_lon"]
            parts.append(df[cols].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def main() -> None:
    train_raw = pd.read_csv(DATA / "train_test.csv")
    val_raw = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv")

    # Everything is fitted on the labelled data only
    cleaner = Cleaner().fit(train_raw)
    train = cleaner.transform(train_raw)
    train["is_outlier"] = outlier_mask(train, fit_rate_baseline(train))
    model = GBMModel(FEATURES, **PARAMS).fit(train)
    print(f"trained on {(~train['is_outlier']).sum():,} of {len(train):,} rows "
          f"({train['is_outlier'].sum()} outliers excluded)")

    # Validation predictions, written in the template's row order
    val = cleaner.transform(val_raw)
    pred = pd.Series(model.predict(val).to_numpy(), index=val["load_id"])
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(pred).round(2)
    assert len(out) == 12_000 and out["load_id"].is_unique
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # December: add coordinates from the city lookup, keep the original seven columns,
    # and only fill predicted_rate
    coords = city_coords(train_raw, val_raw)
    dec_in = dec_raw.copy()
    for side in ["pickup", "delivery"]:
        dec_in[f"{side}_lat"] = dec_in[side].map(coords["lat"])
        dec_in[f"{side}_lon"] = dec_in[side].map(coords["lon"])
    dec = cleaner.transform(dec_in)
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = model.predict(dec).round(2).to_numpy()
    assert dec_out["predicted_rate"].notna().all()
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("wrote validation_predictions.csv and data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
