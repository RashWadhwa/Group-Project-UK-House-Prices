import pandas as pd
import numpy as np
import pytest

from project import load_dataset, filter_all_dwellings, add_time_features


def test_filter_all_dwellings_basic():
    df = pd.DataFrame({
        "Period_dt": pd.to_datetime(["2021-01-01", "2021-04-01"]),
        "Period": ["2021 Q1", "2021 Q2"],
        "Region": ["United Kingdom", "United Kingdom"],
        "Region code": ["K02000001", "K02000001"],
        "All dwellings Price": [345338, 342759],
    })

    out = filter_all_dwellings(df)

    assert set(["Period_dt", "Period", "Region", "Region code", "Price", "Segment"]).issubset(out.columns)
    assert out["Segment"].unique().tolist() == ["All dwellings"]
    assert out["Price"].tolist() == [345338, 342759]


def test_filter_all_dwellings_drops_missing_price():
    df = pd.DataFrame({
        "Period_dt": pd.to_datetime(["2021-01-01", "2021-04-01"]),
        "Period": ["2021 Q1", "2021 Q2"],
        "Region": ["United Kingdom", "United Kingdom"],
        "Region code": ["K02000001", "K02000001"],
        "All dwellings Price": [np.nan, 342759],
    })

    out = filter_all_dwellings(df)
    assert len(out) == 1
    assert out.iloc[0]["Price"] == 342759


def test_add_time_features_creates_lags_and_rolls():
    # Build 6 quarters so lag_4 and roll_mean_4 become valid
    dts = pd.to_datetime([
        "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01"
    ])
    prices = [100, 110, 120, 130, 140, 150]

    hist = pd.DataFrame({
        "Period_dt": dts,
        "Region": ["X"] * 6,
        "Price": prices,
    })

    feats = add_time_features(hist)

    # Check columns exist
    for col in ["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth", "quarter"]:
        assert col in feats.columns

    # On the 5th row (index 4), lag_4 should equal first price
    assert feats.loc[4, "lag_4"] == 100

    # roll_mean_4 at index 4 uses previous 4 prices: 100,110,120,130 => mean 115
    assert feats.loc[4, "roll_mean_4"] == pytest.approx(115.0)

    # yoy_growth at index 4 = price(140)/lag_4(100) - 1 = 0.4
    assert feats.loc[4, "yoy_growth"] == pytest.approx(0.4)


def test_load_dataset_replaces_x_and_parses_date(tmp_path):
    # Minimal CSV to test [x] handling and Period_dt parsing
    p = tmp_path / "sample.csv"
    p.write_text(
        "Period,Region,Region code,All dwellings Price,Period_dt\n"
        "2024 Q1,United Kingdom,K02000001,[x],01/01/2024\n"
        "2024 Q2,United Kingdom,K02000001,327235,01/04/2024\n",
        encoding="utf-8"
    )

    df = load_dataset(str(p))
    assert pd.api.types.is_datetime64_any_dtype(df["Period_dt"])
    assert pd.isna(df.loc[0, "All dwellings Price"])
    assert float(df.loc[1, "All dwellings Price"]) == 327235.0

