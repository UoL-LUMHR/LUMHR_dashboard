# Explaining the SAMHI machine-learning results

Run the chart generator from this directory:

```bash
python 04_plot_results.py --scope both
```

PNG files are written to `results/plots/`. Use `--scope lincolnshire` or
`--scope national` to generate one geography only, and `--output-dir` to choose
a different destination.

## What the graphs show

### 1. Model performance

`01_model_performance_<scope>.png` compares models using RMSE and R². RMSE is
the typical size of the prediction error in SAMHI units, so lower is better.
R² describes how much variation in observed SAMHI is explained by the model,
so higher is better. The chart uses the 2020–2022 test rows when those rows are
available; this is the most demanding out-of-time period in the experiment.

The persistence model is an important reference point: it assumes next year
will equal the previous year. A model is useful only if it improves on this
simple benchmark. Metrics should be compared within the same geography and
test split; Lincolnshire and national scores are not directly interchangeable.

### 2. Observed versus predicted SAMHI

`02_observed_vs_predicted_<scope>.png` shows the mean across LSOAs for each
year. The dark line is observed SAMHI and the other lines are model forecasts.
Close lines indicate good average calibration. This is an aggregate view: a
model can follow the mean well while still making large errors for individual
LSOAs, so read it alongside RMSE and MAE.

The shaded band marks 2020–2022, the COVID-era test period. Forecasts in this
period are evaluated against later observations and should not be interpreted
as causal estimates of the pandemic’s effect.

### 3. Feature importance

`03_feature_importance_<scope>.png` displays the twelve largest mean absolute
ElasticNet LinearSHAP contributions for the primary model. The Flask forecast
page displays the selected model's own driver file: LinearSHAP for Ridge/ElasticNet,
TreeSHAP for tree ensembles, and permutation importance for EBM/stacking. A larger value means that a
feature changed predictions more strongly on average; it does **not** prove
that the feature causes mental-health need. Correlated variables can share or
substitute for one another, and SHAP importance does not show whether an
association is positive or negative.

Temporal variables such as `lag_1` and rolling summaries are expected to be
strong because SAMHI is persistent over time. Structural variables such as
deprivation, disability, isolation, and travel time describe useful context,
but should be treated as predictive signals rather than policy effects.

### 4. Forward projections

`04_forward_projection_summary_<scope>.png` summarises the 2023–2025 recursive
ElasticNet projections. The row-level CSV now also retains projections for the
other benchmark models, including XGBoost, Random Forest, Extra-Trees, CatBoost,
EBM, stacking, and both baselines. The first panel shows mean projected SAMHI; the second shows the
percentage of areas classified as worsening or improving relative to the 2022
baseline. These are model projections, not observed outcomes. Uncertainty is
available in the row-level projection CSV through `ci_lower_95` and
`ci_upper_95`.

## Caveats

The plots summarise the CSV outputs already produced by the modelling scripts;
they do not change model fitting or evaluation. Percentages and averages can
hide variation between LSOAs and districts. Results should therefore support
triage and exploration, not replace local clinical, service-planning, or
data-quality review.
