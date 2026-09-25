# SAMHI history comparison results

This folder contains the completed multimodal SAMHI experiments comparing models that can use previous SAMHI values with models that are deliberately prevented from using them.

## Experiment matrix

Each model was run for:

- Lincolnshire and all England LSOAs.
- The current test window, 2020–2022, trained on 2014–2018 and validated on 2019.
- The pre-COVID test window, 2018–2019, trained on 2014–2016 and validated on 2017.
- `with_history`: includes previous-SAMHI predictors.
- `without_history`: excludes previous-SAMHI predictors.

The comparison includes Ridge, ElasticNet, Random Forest, Extra-Trees, LightGBM, XGBoost, CatBoost, Explainable Boosting Machine, the stacking ensemble, and the two autoregressive reference baselines. The baselines are unchanged between the two modes and are included as reference points.

The new non-SAMHI inputs include fuel poverty, lagged annual gas-grid disconnection, tenure, overcrowding, and detailed household composition. The national fuel-poverty and gas-grid sources supplied in `scripts/utils/source_data/` were used when available. Gas-grid disconnection is lagged by one year. The Census-derived housing and household variables are static covariates repeated across the panel years.

## Main result

Previous SAMHI predictors are useful. Removing them increased test RMSE for every learned model in every geography and time window. The largest effects were generally for the linear models, while the ensemble and tree models also lost substantial accuracy.

Best learned-model RMSE with history:

| Test window | Geography | Best model | RMSE with history |
|---|---|---|---:|
| 2020–2022 | Lincolnshire | Multimodal ElasticNet | 0.2488 |
| 2020–2022 | England | Multimodal ElasticNet | 0.3186 |
| 2018–2019 | Lincolnshire | Multimodal Ridge | 0.2764 |
| 2018–2019 | England | Multimodal Ridge | 0.2968 |

For example, in 2020–2022 the best Lincolnshire ElasticNet RMSE rose from 0.2488 to 0.9147 when history was removed. For England, it rose from 0.3186 to 0.8908. These are predictive-performance comparisons, not evidence that the external variables are unimportant: the no-history models still use the new external features, but lose the local SAMHI temporal and spatial context.

## Files

- `multimodal_metrics_comparison_2020_2022.csv`: combined metrics for the current test window.
- `multimodal_metrics_comparison_pre_covid_2018_2019.csv`: combined metrics for the pre-COVID test window.
- `aggregate_test_metrics.csv`: aggregate test rows used for the plots.
- `history_effect_by_model.csv`: one row per model, geography, and time window. Positive `RMSE_change_without_minus_with` means history improved the model.
- `multimodal_predictions_*.csv`: predictions for each scope and history mode.
- `shap_feature_importance_*.csv`: feature-importance outputs from the benchmark.
- `plots/`: RMSE comparison and history-removal delta plots.

## Reading the plots

In the RMSE comparison plots, lower bars are better. In the history delta plots, positive bars mean that removing previous SAMHI values worsened RMSE, so positive values indicate a benefit from history.

The direction and size of the effect should be interpreted alongside the forecast design. The no-history experiment answers whether the external and other non-SAMHI variables can predict SAMHI without previous SAMHI inputs; it is not a replacement for the autoregressive formulation when the previous SAMHI value will be available at prediction time.

## Reproducing the results

From the repository root:

```bash
scripts/.venv/bin/python scripts/machine_learning/samhi/02_multimodal_features.py \
  --scope both --experiment-set 1 --history both \
  --output-dir scripts/machine_learning/samhi/results/history_comparison

scripts/.venv/bin/python scripts/machine_learning/samhi/02_multimodal_features.py \
  --scope both --experiment-set 2 --history both \
  --output-dir scripts/machine_learning/samhi/results/history_comparison

scripts/.venv/bin/python scripts/machine_learning/samhi/05_plot_history_comparison.py \
  --results-dir scripts/machine_learning/samhi/results/history_comparison
```

The generated metrics are test-set results from the fixed chronological splits. Because the national Census and fuel-poverty inputs are not available as a fully time-varying historical series in this setup, the static variables should be treated as contextual covariates rather than a strict historical-information audit.
