from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project import (
    TrainConfig,
    add_time_features,
    export_plots,
    filter_all_dwellings,
    forecast_next_quarters,
    load_dataset,
    main,
    run_pipeline,
    train_model,
)


def _build_raw_csv(path: Path, periods: int = 12) -> Path:
    rows = []
    start = pd.Timestamp("2021-01-01")
    for i in range(periods):
        dt = start + pd.offsets.QuarterBegin(startingMonth=1) * i
        rows.append(
            {
                "Period": f"{dt.year} Q{dt.quarter}",
                "Region": "United Kingdom",
                "Region code": "K02000001",
                "All dwellings Price": 300000 + (i * 2500),
                "All dwellings average advance": 180000 + (i * 700),
                "All dwellings average recorded income of borrowers": 40000 + (i * 120),
                "Period_dt": dt.strftime("%d/%m/%Y"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _build_features_for_training() -> pd.DataFrame:
    dts = pd.date_range("2021-01-01", periods=12, freq="QS")
    hist = pd.DataFrame(
        {
            "Period_dt": dts,
            "Period": [f"{d.year} Q{d.quarter}" for d in dts],
            "Region": ["United Kingdom"] * len(dts),
            "Region code": ["K02000001"] * len(dts),
            "Price": [300000 + (i * 2500) for i in range(len(dts))],
            "Segment": ["All dwellings"] * len(dts),
        }
    )
    return add_time_features(hist)


def test_load_dataset(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text(
        "Period,Region,Region code,All dwellings Price,Period_dt\n"
        "2024 Q1,United Kingdom,K02000001,[x],01/01/2024\n"
        "2024 Q2,United Kingdom,K02000001,327235,01/04/2024\n",
        encoding="utf-8",
    )

    df = load_dataset(str(p))
    assert pd.api.types.is_datetime64_any_dtype(df["Period_dt"])
    assert pd.isna(df.loc[0, "All dwellings Price"])
    assert float(df.loc[1, "All dwellings Price"]) == 327235.0


def test_filter_all_dwellings():
    df = pd.DataFrame(
        {
            "Period_dt": pd.to_datetime(["2021-01-01", "2021-04-01"]),
            "Period": ["2021 Q1", "2021 Q2"],
            "Region": ["United Kingdom", "United Kingdom"],
            "Region code": ["K02000001", "K02000001"],
            "All dwellings Price": [np.nan, 342759],
        }
    )

    out = filter_all_dwellings(df)
    assert set(["Period_dt", "Period", "Region", "Region code", "Price", "Segment"]).issubset(out.columns)
    assert out["Segment"].unique().tolist() == ["All dwellings"]
    assert len(out) == 1
    assert out.iloc[0]["Price"] == 342759


def test_add_time_features():
    dts = pd.to_datetime(
        ["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01"]
    )
    prices = [100, 110, 120, 130, 140, 150]

    hist = pd.DataFrame(
        {
            "Period_dt": dts,
            "Region": ["X"] * 6,
            "Price": prices,
        }
    )

    feats = add_time_features(hist)

    for col in ["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth", "quarter"]:
        assert col in feats.columns
    assert feats.loc[4, "lag_4"] == 100
    assert feats.loc[4, "roll_mean_4"] == pytest.approx(115.0)
    assert feats.loc[4, "yoy_growth"] == pytest.approx(0.4)


def test_train_model():
    feats = _build_features_for_training()
    model, metrics = train_model(feats, TrainConfig())

    assert hasattr(model, "predict")
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
    assert metrics["mae"] >= 0


def test_forecast_next_quarters():
    feats = _build_features_for_training()
    model, _ = train_model(feats, TrainConfig())

    fc = forecast_next_quarters(model, feats, n_quarters=2)
    assert not fc.empty
    assert set(["Period_dt", "Region", "Segment", "yhat"]).issubset(fc.columns)


def test_export_plots(tmp_path):
    feats = _build_features_for_training()
    model, metrics = train_model(feats, TrainConfig())

    model_feature_cols = ["lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth", "quarter", "Region"]
    frame = feats.dropna(subset=["Price", "lag_1", "lag_2", "lag_4", "roll_mean_4", "yoy_growth"]).copy()
    frame["rank_desc"] = frame.groupby("Region")["Period_dt"].rank(method="first", ascending=False)
    test_frame = frame.loc[frame["rank_desc"] <= TrainConfig().test_quarters].copy()
    x_test = test_frame[model_feature_cols]
    y_test = test_frame["Price"]
    y_pred = model.predict(x_test)

    history = feats.loc[:, ["Period_dt", "Region", "Price", "Segment"]].copy()
    forecast = forecast_next_quarters(model, feats, n_quarters=2)

    reports_dir = tmp_path / "reports"
    export_plots(metrics, history, forecast, model, x_test, y_test, y_pred, reports_dir)

    assert (reports_dir / "actual_vs_predicted.png").exists()
    assert (reports_dir / "history_and_forecast.png").exists()
    assert (reports_dir / "feature_importances.png").exists()


def test_run_pipeline(tmp_path):
    csv_path = _build_raw_csv(tmp_path / "five_year_dataset.csv", periods=12)
    output_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"

    results = run_pipeline(str(csv_path), output_dir=output_dir, reports_dir=reports_dir)

    assert (output_dir / "clean_prices_all_dwellings.csv").exists()
    assert (output_dir / "features_all_dwellings.csv").exists()
    assert (output_dir / "forecast_all_dwellings_12q.csv").exists()
    assert (reports_dir / "actual_vs_predicted.png").exists()
    assert "metrics" in results


def test_main(monkeypatch):
    monkeypatch.setattr("sys.argv", ["project.py"])
    with pytest.raises(SystemExit):
        main()

