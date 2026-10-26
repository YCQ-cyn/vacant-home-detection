# Private data contract

[中文](../zh/data_contract.md) | **English**

All paths below are relative to `VACANT_HOME_WORKSPACE/recognition`. No corresponding data files are shipped in the repository. Trusted local pickle/joblib files must not be obtained from unknown sources.

## Prepared training inputs

| Path | Meaning |
| --- | --- |
| `inputs/power_samples.xlsx` | Raw workbook; `Sheet1` with meter/utility/hardware metadata and clock columns |
| `inputs/legacy_labels.xlsx` | Historical expert positives: first column meter ID, second column subtype label |
| `inputs/training_scope.csv` | Per-meter eligibility, historical provenance, protected-cohort routing |
| `inputs/data_snapshot/daily_curves.pkl` | Clean daily frame, meter/date, 96 clock columns and quality metadata |
| `inputs/semantics_robust_core.csv` | Prepared per-meter semantics-robust behavioral feature table |
| `inputs/automatic_human_axes.csv` | Existing axes used only by the pre-fit review-pool scout |
| `inputs/scout_screening.csv` | Frozen scout results used to design the protected review pool |
| `inputs/daily_inputs.npz` | Float32 `x`: daily rows × 8 channels × 96 samples |
| `inputs/daily_input_index.csv` | Meter/date rows aligned exactly with `x` |

Training requires the complete prepared input set, including features, provenance, labels and preliminary screening results. Raw-to-feature preprocessing is outside this package's interface. Source-only installations support tests and method inspection.

## Daily curves

Columns: `meter_id`, `date`, `utility`, `0:00`, `0:15`, …, `23:45`. Quality columns include `unresolved_missing_points` and `original_missing_points`. A possible `24:00` source column is not an extra feature sample; the encoder consumes 96 points.

Channels, in order:

1. Signed-log absolute power.
2. Robust normalized daily shape.
3. Signed-log sample difference.
4. Relative sample difference.
5. Original missing mask.
6. Interpolation mask.
7. Unresolved missing mask.
8. Negative sample mask.

Keep masked values finite in the prepared encoder tensor; retain the masks instead of silently treating missingness as valid electrical silence.

## Scope and protection

`training_scope.csv` fields: `meter_id`, `data_invalid`, `strict_all_zero`, `model_a_fit`, `shadow_fit_excluding_gold_test`, `literal_transductive_full_fit`, `legacy_high_hard`, `legacy_medium_soft`, `legacy_low_diagnostic_only`, `protected_cohort`.

Preparation creates `fit_scope.csv`, `future_human_holdout_manifest.csv` and `future_human_holdout_freeze.json`. The input contract requires 30 held users, split 10 random blind / 20 targeted challenge. Freeze these before SSL or any reference/fit. Existing protected holdouts must remain unchanged.

## Inference inputs and outputs

`final_fit_feature_table.parquet` contains `meter_id` plus the fitted robust and embedding columns. `models/final_pu_channels.joblib` stores per-channel estimators and frozen rank references. `models/conditional_subtype.joblib` stores subtype preprocessing/classifiers/prototypes. A compatible fitted model is supplied locally.

`infer` outputs the selected PU evidence score/rank and four conditional subtype supports. The full training validation/report flow adds routers, independent axes, consensus, conflict, stability and the business screening classes.

The specialist branch requires local final embeddings, `screening_results.csv`, fit scope, protected holdout manifest and expert positives. All outputs go to `VACANT_HOME_WORKSPACE/high_power`.
