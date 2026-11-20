# Lincolnshire boundary aggregation comparison

Both runs use the same current feature set, train/test years, model settings and random seeds. The only comparison switch is how 2021 geography is represented on 2011 model areas. Duplicate feature columns were removed before both runs.

## Geography and aggregation

- The model retains 420 2011 LSOAs and the same 1,260 test rows (420 areas × 3 years).
- In this Lincolnshire lookup, 406 areas map one-to-one; 13 split into two 2021 children and one splits into three.
- Household rates use 2021 household counts; QOF rates use registered-patient counts; rurality uses the population-weighted majority; IMD ranks/deciles use a population-weighted mean. The IMD summaries are not newly calculated parent-area IMD scores.
- Gas-grid percentages use household counts as a proxy weight because the model input table does not contain property denominators.
- The 2021 child adjacency graph is collapsed to 2011 parent areas before calculating spatial lags.

## Held-out performance (2020–2022)

| Model | First child RMSE | Aggregated RMSE | Change |
|---|---:|---:|---:|
| ElasticNet | 0.248753 | 0.248532 | −0.000221 |
| Ridge | 0.260327 | 0.260209 | −0.000118 |
| Random Forest | 0.274363 | 0.274276 | −0.000087 |
| Extra-Trees | 0.278858 | 0.275874 | −0.002984 |
| LightGBM | 0.283480 | 0.281820 | −0.001660 |
| XGBoost | 0.282334 | 0.283855 | +0.001521 |
| CatBoost | 0.292459 | 0.293322 | +0.000863 |
| EBM | 0.280912 | 0.282285 | +0.001373 |
| Stacking | 0.259058 | 0.259450 | +0.000392 |

ElasticNet remains the best-performing model. Its RMSE changes by about 0.09%; results are mixed across the other models.

## Explanations and predictions

For LightGBM, SHAP domain shares change by less than 0.51 percentage points; for ElasticNet, by less than 0.23 points. ElasticNet test predictions change by a mean absolute 0.00226 SAMHI units (maximum 0.0762). LightGBM predictions change by a mean absolute 0.0248 units (maximum 0.1286), while its overall RMSE improves by 0.00166.

Comparison files:

- `boundary_aggregation_comparison_metrics.csv`
- `boundary_aggregation_comparison_shap_domains.csv`
- `boundary_aggregation_prediction_changes.csv`

These are Lincolnshire results only. The national exact-fit lookup also contains merged/complex relationships, where a 2021 feature cannot always be uniquely recovered for each 2011 area.
