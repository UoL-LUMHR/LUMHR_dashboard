# Lincolnshire boundary aggregation comparison

Both runs use the same current feature set, train/test years, model settings and random seeds. The only comparison switch is how 2021 geography is represented on 2011 model areas. Duplicate feature columns were removed before both runs.

## Geography and aggregation

- The model retains 420 2011 LSOAs and the same 1,260 test rows (420 areas × 3 years).
- In this Lincolnshire lookup, 406 areas map one-to-one; 13 split into two 2021 children and one splits into three.
- Household rates use 2021 household counts; QOF rates use registered-patient counts; rurality uses the population-weighted majority; IMD ranks/deciles use a population-weighted mean. The IMD summaries are not newly calculated parent-area IMD scores.
- The corrected 2021 RUC feature records each parent area's population share in children classified as nearer to a major town or city; the obsolete six-level isolation score was removed.
- Gas-grid percentages use household counts as a proxy weight because the model input table does not contain property denominators.
- The 2021 child adjacency graph is collapsed to 2011 parent areas before calculating spatial lags.

## Held-out performance (2020–2022)

| Model | First child RMSE | Aggregated RMSE | Change |
|---|---:|---:|---:|
| ElasticNet | 0.248753 | 0.248532 | −0.000221 |
| Ridge | 0.260215 | 0.260106 | −0.000109 |
| Random Forest | 0.274231 | 0.274890 | +0.000660 |
| Extra-Trees | 0.277214 | 0.275158 | −0.002057 |
| LightGBM | 0.282215 | 0.281963 | −0.000253 |
| XGBoost | 0.284826 | 0.287535 | +0.002709 |
| CatBoost | 0.290802 | 0.291197 | +0.000395 |
| EBM | 0.283409 | 0.283981 | +0.000572 |
| Stacking | 0.259682 | 0.259231 | −0.000451 |

ElasticNet remains the best-performing model. Its RMSE changes by about 0.09%; results are mixed across the other models.

## Explanations and predictions

For LightGBM, SHAP domain shares change by at most 1.24 percentage points; for ElasticNet, by at most 0.23 points. ElasticNet test predictions change by a mean absolute 0.00226 SAMHI units (maximum 0.0762). LightGBM predictions change by a mean absolute 0.0251 units (maximum 0.1549), while its overall RMSE improves by 0.00025.

## Corrected RUC feature

The old 2021 label-to-scale mapping returned the default value for every area. It has been replaced with a population share based on the 2021 RUC nearer/further codes. In the first-child sensitivity run, the new feature has 0.04% LightGBM SHAP importance and 0% ElasticNet SHAP importance. ElasticNet RMSE is unchanged; other models show small, mixed changes. The feature is now valid, but it is not a major driver of this forecast.

Comparison files:

- `boundary_aggregation_comparison_metrics.csv`
- `boundary_aggregation_comparison_shap_domains.csv`
- `boundary_aggregation_prediction_changes.csv`

These are Lincolnshire results only. The national exact-fit lookup also contains merged/complex relationships, where a 2021 feature cannot always be uniquely recovered for each 2011 area.
