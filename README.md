# Freight Rate Prediction

Predicts the posted rate for a truckload from its lane, distance, equipment, weight and date.
Trained on Jan–Oct 2025 loads (`data/train_test.csv`), predicts the 12,000 Nov–Dec 2025 loads in `data/validation.csv`.

## How to run

Python 3.10+.

```bash
python -m pip install -r requirements.txt

# 1. Train the final model and write both prediction files
python src/final.py

# 2. Validate the files and draw the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`src/final.py` takes a few seconds and writes:

| File | What it is |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | the December input file with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | December chart (written by `score.py`) |

The notebooks show how I got to the final model. Open them with `jupyter notebook` from the `notebooks/` folder and read them in order.

## Project layout

```
data/                 input files from the assessment
notebooks/
  01_eda              what's in the data and what's wrong with it
  02_cleaning         cleaning rules and checks
  03_split            time-based folds and the unseen-city test
  04_baselines        simple rate-per-mile rules to beat
  05_features         which feature groups help, fold by fold
  06_models           model and settings choice
  07_final_checks     sanity checks on the final predictions
src/
  clean.py            Cleaner (fit on train only) + bad-rate flag
  splits.py           time folds; cleaning is refit inside each fold
  cv.py, metrics.py   evaluation helpers
  baselines.py        baseline models
  features.py         feature groups
  gbm.py, linear.py   models
  final.py            trains on all data, writes the submission files
report/               write-up: split and validation approach, results, December chart
score.py              scorer provided with the assessment (unchanged)
```

## Approach in short

**Split.** Validation is the two months after training ends, so I validate the same way: three expanding-window folds (Jan–Apr → May–Jun, Jan–Jun → Jul–Aug, Jan–Aug → Sep–Oct). Cleaning and outlier flags are refit inside each fold on its training rows only. A second test holds 8 random cities out of training, because validation has 8 cities the model has never seen.

**Cleaning.**
- Negative weights (292) are sign flips, so I take the absolute value.
- Weights stuck at 47,500 (capped) or 5,000 (floored) are kept, with a flag.
- Missing weights get the median for that equipment type, plus a flag.
- About 1.4% of rates (677 loads) are far outside the normal rate per mile for their equipment and distance, in two separate clusters. They look like entry errors, so they're excluded from training.

**Model.** Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) on log(rate per mile), with distance, equipment, weight, day of week and pickup/delivery coordinates. Coordinates are what let it price the 8 new cities. `market_index` and `quote_signal` were tested but left out, because they help in some months and hurt badly in others (details in `05_features.ipynb`).

**Results** (average error in dollars on normal loads, Sep–Oct fold, the closest to the real task):

| Model | MAE |
|---|---|
| Distance × one global rate per mile | $204 |
| Best baseline (lane × equipment medians) | $77 |
| Final model | **$42** |

On the unseen-city test, the final model's error is $48, against $63 without coordinates.
