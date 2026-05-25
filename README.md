
# Group-Project-UK-House-Prices

## Overview

This project builds a simple UK house-price forecasting pipeline from an ONS-style CSV:

- cleans and standardizes the source data,
- engineers time-series features by region,
- trains a tree-based regression model,
- evaluates on a time-based holdout,
- forecasts the next 12 quarters (3 years),
- exports CSV outputs and report images.

The single source of truth is `project.py`.

## Requirements

- Python 3.9+
- packages listed in `requirements.txt`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

Development dataset:

- `data/five_year_dataset.csv`

Source dataset (ONS):

- https://www.ons.gov.uk/peoplepopulationandcommunity/housing#datasets

Project note:

- The CSV file `data/five_year_dataset.csv` was created by extracting the latest 5 years of data from the ONS housing datasets link above, and this extracted file is the one used throughout this project.

Expected key columns include:

- `Period_dt`
- `Period`
- `Region`
- `Region code`
- `All dwellings Price`

## How To Run

Primary workflow (recommended for deployment and GitHub use):

```bash
python project.py data/five_year_dataset.csv
```

Optional notebook workflow:

- `docs/training_and_plotting.ipynb` now calls the same pipeline function from `project.py`.
- Use the notebook only if you want interactive preview tables/metrics.
- You do not need to run both script and notebook.

## Outputs

After running, the pipeline writes:

- cleaned history: `outputs/clean_prices_all_dwellings.csv`
- engineered features: `outputs/features_all_dwellings.csv`
- 12-quarter forecast: `outputs/forecast_all_dwellings_12q.csv`

And report images to:

- `docs/reports/actual_vs_predicted.png`
- `docs/reports/history_and_forecast.png`
- `docs/reports/feature_importances.png`

## Model Summary

Features used for training:

- `lag_1`, `lag_2`, `lag_4`
- `roll_mean_4`
- `yoy_growth`
- `quarter`
- `Region` (one-hot encoded)

Model:

- `HistGradientBoostingRegressor`
- wrapped in an `sklearn` `Pipeline` with `ColumnTransformer` and `OneHotEncoder(handle_unknown="ignore")`

Validation:

- time-based holdout per region using the last 4 quarters as test data
- reports `mae`, `rmse`, `mape`, plus naive baseline metrics

## Tests

Run tests with:

```bash
pytest -q
```

Unit tests are defined in `test_project.py`.

## Repository Notes

- keep one reports folder only: `docs/reports/`
- `project.py` is the production entrypoint
- notebook is optional and mirrors the same pipeline
