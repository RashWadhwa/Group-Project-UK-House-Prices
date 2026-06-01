Sklearn & Pipeline Reference
===========================

Purpose
-------
This document explains the main libraries and sklearn components used in this project, why a histogram-gradient boosting regressor was chosen, and how to produce a PDF reference from this Markdown file.

Data source
-----------
The input data for this project is an Office for National Statistics (ONS) style extract. For convenience, the repository includes the latest five years of regional house-price data used for development: data/five_year_dataset.csv. The CSV contains expected ONS-style columns such as Period_dt, Period, Region, Region code, and All dwellings Price. The pipeline normalises common ONS markers (for example, [x]) to NaN and parses dates with dayfirst=True.

Key libraries used
------------------
- pandas — data loading and tabular manipulation. Used to read CSVs, parse Period_dt, and build the tidy tables written to outputs/.
- numpy — numeric arrays and simple numerical utilities (NaN handling, means, basic arithmetic).
- scikit-learn (sklearn) — modelling and preprocessing building blocks used in project.py:
  - sklearn.compose.ColumnTransformer — applies different preprocessing to subsets of columns (here: one-hot encoding of the Region column, while passing numeric features through unchanged). Keeps preprocessing logic tidy and compatible with Pipeline.
  - sklearn.pipeline.Pipeline — chains preprocessing + estimator. Ensures the same transforms are applied at training and prediction time and provides a single .fit()/.predict() interface.
  - sklearn.preprocessing.OneHotEncoder — converts categorical Region values into binary indicator columns. The project uses handle_unknown="ignore" so the pipeline can safely predict on regions not seen during training.
  - sklearn.metrics.mean_absolute_error and sklearn.metrics.mean_squared_error — compute MAE and MSE; RMSE is computed as sqrt(MSE). These are standard, interpretable regression metrics used for model evaluation and baseline comparison.
  - sklearn.ensemble.HistGradientBoostingRegressor — the chosen estimator. See the section below for motivation.

Why HistGradientBoostingRegressor (HGBR)?
------------------------------------------
Short answer: strong, efficient default for tabular regression tasks with heterogeneous features.

Rationale and benefits:

- Tree-based model: handles nonlinear relationships and interactions without feature scaling; works well when predictors include lags, rolling aggregates, and categorical encodings.
- Histogram-based algorithm: HGBR bins continuous features into histograms before splitting. This often delivers large speedups on medium-to-large tabular datasets compared to classical gradient boosting implementations, while maintaining similar accuracy.
- Performance and memory: HGBR is optimised in scikit-learn and tends to be faster and more memory-efficient than the older GradientBoosting implementation for many datasets.
- Usability: part of sklearn core — no external dependencies (unlike XGBoost/LightGBM) and integrates seamlessly with Pipeline/ColumnTransformer.
- Robust baseline: tends to give strong baseline results on tabular data; a good first-choice model when you want decent accuracy quickly.

When to consider alternatives
----------------------------
- If you need extreme speed or extra tuning knobs, LightGBM or XGBoost may outperform HGBR on large datasets.
- If interpretability is a priority, consider LinearRegression (for simple linear relationships) or add SHAP/explainability tooling for tree ensembles.
- If your data contains many missing values and you'd like native handling, check the specific sklearn version's behaviour; some tree implementations treat NaN specially while others require imputation.

How the model is used in this project
------------------------------------
1. Feature engineering (in add_time_features): lags (lag_1, lag_2, lag_4), rolling mean (roll_mean_4), year-over-year growth (yoy_growth), and quarter.
2. Preprocessing: ColumnTransformer one-hot encodes Region; numeric features are passed through unchanged.
3. Training: a time-based per-region holdout is used (last cfg.test_quarters per region are test rows). The pipeline is fit for the training portion.
4. Baseline: naive forecast uses lag_1. Model metrics (MAE, RMSE, MAPE) are compared to the baseline to assess improvement.
5. Forecasting: iterative per-region predictions where the predicted yhat values are appended to the region's series and used to compute features for subsequent quarters.

Recommended evaluation practices
--------------------------------
- Keep the time-based holdout per group (region) — avoids leakage and simulates real forecasting.
- Report both absolute errors (MAE, RMSE) and relative error (MAPE) since house prices vary widely by region.
- Compare to a simple baseline (lag-1) to check whether the model is adding predictive value.

Lags — definition and purpose
----------------------------
- Definition: a lag-k is the value of the series k time steps before the current period. If Price[t] is the price at time `t`, then `lag_1[t] = Price[t-1]`, `lag_4[t] = Price[t-4]`, etc.
- Purpose: lags capture autocorrelation and seasonal behaviour. For quarterly house prices, lag_1 often captures immediate momentum and lag_4 captures year-on-year effects.
- Implementation note: compute lags per-region using groupby('Region').shift(k) (as in add_time_features) to avoid mixing series across regions.


