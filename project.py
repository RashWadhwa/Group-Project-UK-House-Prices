import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


SEGMENT_ALL_DWELLINGS = "All dwellings"


@dataclass(frozen=True)
class TrainConfig:
    horizon_quarters: int = 12  # 3 years
    test_quarters: int = 4      # last 1 year held out
    random_state: int = 42


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw ONS-like CSV and do minimal normalization:
    - Parse Period_dt
    - Replace [x] with NaN
    - Convert known numeric columns to numeric when possible
    """
    df = pd.read_csv(path)

    if "Period_dt" not in df.columns:
        raise ValueError("Expected a 'Period_dt' column in the dataset.")

    # Strip whitespace in text fields so " [x]" etc. are handled
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Normalize missing markers commonly used in ONS extracts
    df = df.replace({"[x]": np.nan, "x": np.nan, "X": np.nan, "": np.nan})

    # Parse date (your dataset uses dd/mm/yyyy like 01/10/2025)
    df["Period_dt"] = pd.to_datetime(df["Period_dt"], dayfirst=True, errors="coerce")
    if df["Period_dt"].isna().any():
        bad = df.loc[df["Period_dt"].isna(), "Period"].head(5).tolist()
        raise ValueError(f"Some Period_dt values could not be parsed. Examples Period={bad}")

    # Convert key numeric columns (others may exist; keep it focused)
    numeric_cols = [
        "All dwellings Price",
        "All dwellings average advance",
        "All dwellings average recorded income of borrowers",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_all_dwellings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean history table for All dwellings only.
    Columns returned are standardized for downstream steps.
    """
    required = {"Period_dt", "Period", "Region", "Region code", "All dwellings Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[:, ["Period_dt", "Period", "Region", "Region code", "All dwellings Price"]].copy()
    out = out.rename(columns={"All dwellings Price": "Price"})
    out["Segment"] = SEGMENT_ALL_DWELLINGS

    # Sort and drop rows where target is missing
    out = out.sort_values(["Region", "Period_dt"]).reset_index(drop=True)
    out = out.dropna(subset=["Price"]).reset_index(drop=True)
    return out


def add_time_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series features per region:
    - lags: 1,2,4
    - rolling mean over last 4 quarters (excluding current)
    - yoy growth based on lag_4
    - quarter (1..4)
    """
    required = {"Period_dt", "Region", "Price"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = history.copy()
    df = df.sort_values(["Region", "Period_dt"]).reset_index(drop=True)

    g = df.groupby("Region", group_keys=False)

    df["lag_1"] = g["Price"].shift(1)
    df["lag_2"] = g["Price"].shift(2)
    df["lag_4"] = g["Price"].shift(4)

    # rolling mean of previous 4 quarters (shift first so it doesn't leak current)
    df["roll_mean_4"] = g["Price"].shift(1).rolling(window=4, min_periods=4).mean()

    df["yoy_growth"] = (df["Price"] / df["lag_4"]) - 1.0
    df["quarter"] = df["Period_dt"].dt.quarter.astype(int)

    return df


def train_model(features: pd.DataFrame, cfg: TrainConfig) -> tuple[Pipeline, dict]:
    """
    Train a model to predict Price using engineered features.
    Uses last cfg.test_quarters per region as test set (time-based holdout).
    Returns fitted pipeline + metrics dict.
    """
    df = features.copy()

    # We need rows where all model features exist (avoid NaNs from early lags)
    model_feature_cols = ["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth", "quarter", "Region"]
    needed = set(model_feature_cols + ["Price", "Period_dt"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for training: {sorted(missing)}")

    df = df.dropna(subset=["Price", "lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth"]).copy()

    # Create a per-region time split: last N quarters for each region are test
    df["rank_desc"] = df.groupby("Region")["Period_dt"].rank(method="first", ascending=False)
    test_mask = df["rank_desc"] <= cfg.test_quarters
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df[model_feature_cols]
    y_train = train_df["Price"]
    X_test = test_df[model_feature_cols]
    y_test = test_df["Price"]

    pre = ColumnTransformer(
        transformers=[
            ("region", OneHotEncoder(handle_unknown="ignore"), ["Region"]),
        ],
        remainder="passthrough",
    )

    model = HistGradientBoostingRegressor(random_state=cfg.random_state)

    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # Predictions
    pred = pipe.predict(X_test)

    # Baseline: naive = lag_1 (last quarter's price)
    baseline = X_test["lag_1"].to_numpy()

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mape": float(np.mean(np.abs((y_test.to_numpy() - pred) / y_test.to_numpy()))),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "baseline_mape": float(np.mean(np.abs((y_test.to_numpy() - baseline) / y_test.to_numpy()))),
    }

    return pipe, metrics


def forecast_next_quarters(
    model: Pipeline,
    history_with_features: pd.DataFrame,
    n_quarters: int = 12,
) -> pd.DataFrame:
    """
    Iteratively forecast future Price for each Region for n_quarters ahead.
    Uses the same feature columns as training.

    Returns a tidy forecast table:
      Period_dt, Region, Segment, yhat
    """
    df = history_with_features.copy()
    df = df.sort_values(["Region", "Period_dt"]).reset_index(drop=True)

    feature_cols = ["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth", "quarter", "Region"]

    # We will forecast region-by-region to keep logic clear.
    forecasts = []

    for region, region_hist in df.groupby("Region", sort=False):
        region_hist = region_hist.sort_values("Period_dt").reset_index(drop=True)

        # Start from the last observed date
        last_dt = region_hist["Period_dt"].max()

        # We need a working series of prices (actual + predicted)
        prices = region_hist["Price"].astype(float).tolist()
        dts = region_hist["Period_dt"].tolist()

        for i in range(1, n_quarters + 1):
            next_dt = (pd.Timestamp(last_dt) + pd.offsets.QuarterBegin(startingMonth=1) * i)
            # QuarterBegin(startingMonth=1) advances to quarter starts aligned with Jan/Apr/Jul/Oct.

            # Compute features from available prices
            # lags based on the last known points in `prices`
            def get_lag(k: int):
                return prices[-k] if len(prices) >= k else np.nan

            lag_1 = get_lag(1)
            lag_2 = get_lag(2)
            lag_4 = get_lag(4)

            # rolling mean of last 4 quarters excluding "current" => last 4 known prices
            roll_mean_4 = np.mean(prices[-4:]) if len(prices) >= 4 else np.nan
            yoy_growth = (lag_1 / lag_4) - 1.0 if (lag_4 is not None and not np.isnan(lag_4) and lag_4 != 0) else np.nan
            quarter = int(next_dt.quarter)

            row = pd.DataFrame([{
                "Region": region,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_4": lag_4,
                "roll_mean_4": roll_mean_4,
                "yoy_growth": yoy_growth,
                "quarter": quarter,
            }])

            # If early history is too short, model features might be NaN; stop forecasting for this region
            if row[["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth"]].isna().any(axis=None):
                # In your dataset this won't happen because you have >= 5 years,
                # but we keep it safe.
                break

            yhat = float(model.predict(row[feature_cols])[0])

            prices.append(yhat)
            dts.append(next_dt)

            forecasts.append({
                "Period_dt": next_dt,
                "Region": region,
                "Segment": SEGMENT_ALL_DWELLINGS,
                "yhat": yhat,
            })

    return pd.DataFrame(forecasts)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python project.py five_year_dataset.csv")

    input_path = sys.argv[1]
    cfg = TrainConfig()

    raw = load_dataset(input_path)
    all_dw = filter_all_dwellings(raw)
    feats = add_time_features(all_dw)

    model, metrics = train_model(feats, cfg)

    # Forecast from last observed quarter for 12 quarters (3 years)
    fc = forecast_next_quarters(model, feats, n_quarters=cfg.horizon_quarters)

    # Outputs for Power BI / portfolio
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    all_dw.to_csv(outdir / "clean_prices_all_dwellings.csv", index=False)
    feats.to_csv(outdir / "features_all_dwellings.csv", index=False)
    fc.to_csv(outdir / "forecast_all_dwellings_12q.csv", index=False)

    # Print a clean console summary
    print("Model evaluation (time holdout):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    last_hist = all_dw.groupby("Region")["Period_dt"].max().min()
    print(f"\nForecast written to outputs/forecast_all_dwellings_12q.csv")
    print(f"Forecast horizon: {cfg.horizon_quarters} quarters (next 3 years) from last observed quarter.")
    print(f"Note: earliest region last-observed date (sanity check) = {last_hist.date()}")


if __name__ == "__main__":
    main()
