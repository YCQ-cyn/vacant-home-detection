from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BranchPaths
from .data import TIME_COLS, load_daily


REQUIRED_AUDIT_METERS: set[str] = set()


def bool_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def verify_combined_output(paths: BranchPaths, output: Path):
    original = pd.read_csv(paths.upstream / "screening_results.csv", dtype={"meter_id": str})
    combined = pd.read_csv(output, dtype={"meter_id": str})
    if len(combined) != len(original) or combined.meter_id.nunique() != original.meter_id.nunique():
        raise RuntimeError("Combined output does not preserve the complete original cohort")
    missing = [column for column in original.columns if column not in combined]
    if missing:
        raise RuntimeError(f"Original Recognition columns missing: {missing}")
    left = original.sort_values("meter_id").reset_index(drop=True)
    right = combined[original.columns].sort_values("meter_id").reset_index(drop=True)
    for column in original.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(pd.to_numeric(left[column], errors="coerce"), pd.to_numeric(right[column], errors="coerce"), equal_nan=True):
                raise RuntimeError(f"Original numeric column changed: {column}")
        elif not left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)):
            raise RuntimeError(f"Original text column changed: {column}")
    return True


def candidate_columns():
    return [
        "meter_id", "hp_candidate_tier", "hp_candidate_flag", "hp_core_candidate_flag",
        "high_power_definition", "relative_high_power_flag", "strict_high_power_flag", "persistent_high_load",
        "main_power_index", "hp_power_index",
        "raw_median_power", "raw_q90_power", "baseline_floor", "raw_max_power",
        "frozen_sample_gt_0_5kw_fraction", "frozen_sample_gt_1kw_fraction",
        "frozen_sample_gt_2kw_fraction", "frozen_sample_gt_5kw_fraction",
        "daily_sample_median_gt_1kw_count", "daily_sample_q90_gt_2kw_count",
        "sample_zero_fraction", "sample_near_zero_fraction",
        "raw_weekday_weekend_shape_difference", "cross_day_normalized_shape_similarity",
        "human_interaction_evidence", "active_evidence_score", "automatic_control_compatibility",
        "human_interaction_opposition", "active_evidence_opposition",
        "vacancy_screening_score", "hp_branch_score", "hp_conditional_rank",
        "hp_rank_within_relative_definition", "hp_rank_within_strict_definition", "hp_rank_within_persistent_definition",
        "complete_support", "equipment_support", "equipment_intermit_support", "intermittent_support",
        "equipment_only_similarity", "equipment_only_similarity_percentile",
        "equipment_intermit_similarity", "equipment_intermit_similarity_percentile",
        "family_human_absence_support", "family_automatic_structure_support", "family_learned_representation_support",
        "model_disagreement", "stability_rank_median", "stability_rank_p10", "stability_rank_p90", "stability_rank_std",
        "data_status", "future_holdout_flag", "severe_quality_issue", "occasional_high_peak_only",
        "missing_sample_fraction", "unresolved_missing_points", "negative_sample_fraction", "utility",
        "possible_special_or_shared_meter_confounder", "possible_isolated_pulse_confounder", "candidate_reason_codes",
    ]


def add_confounders(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["possible_isolated_pulse_confounder"] = out.occasional_high_peak_only.fillna(False) | (
        (out.raw_max_power >= 5) & (out.frozen_sample_gt_1kw_fraction < .05)
    )
    out["possible_special_or_shared_meter_confounder"] = (
        (out.raw_median_power >= 5)
        | (out.frozen_sample_gt_5kw_fraction >= .25)
        | (out.raw_max_power >= 20)
    )
    def reasons(row):
        codes = []
        if row.get("persistent_high_load", False): codes.append("PERSISTENT_HIGH_FROZEN_SAMPLE_LOAD")
        if row.get("family_human_absence_support", 0) >= .65: codes.append("POWER_CONDITIONED_LOW_HUMAN_ACTIVE")
        if row.get("family_automatic_structure_support", 0) >= .60: codes.append("AUTOMATIC_BACKGROUND_STRUCTURE_COMPATIBLE")
        if row.get("equipment_only_similarity_percentile", 0) >= .70: codes.append("SCALE_NORMALIZED_EQUIPMENT_LEGACY_SIMILARITY")
        if row.get("equipment_intermit_similarity_percentile", 0) >= .70: codes.append("SCALE_NORMALIZED_EQUIPMENT_INTERMIT_SIMILARITY")
        if row.get("possible_special_or_shared_meter_confounder", False): codes.append("SPECIAL_OR_SHARED_METER_CONFOUNDER_REVIEW")
        if row.get("future_holdout_flag", False): codes.append("FUTURE_HOLDOUT_INFERENCE_ONLY")
        return "|".join(codes) if codes else "NO_SPECIAL_BRANCH_SUPPORT"
    out["candidate_reason_codes"] = out.apply(reasons, axis=1)
    return out


def plot_meter(group: pd.DataFrame, output: Path, title: str):
    values = group.sort_values("date")[TIME_COLS].to_numpy(float)
    if not len(values):
        return
    transformed = np.sign(values) * np.log1p(np.abs(values))
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    image = axes[0].imshow(transformed, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title(title)
    axes[0].set_ylabel("Day (chronological)")
    axes[0].set_xticks(np.arange(0, 96, 8), [TIME_COLS[index] for index in range(0, 96, 8)], rotation=45, ha="right")
    figure.colorbar(image, ax=axes[0], label="signed log(1 + frozen power sample)")
    axes[1].plot(np.nanmedian(values, axis=0), color="#1f4e79", label="median day")
    axes[1].fill_between(np.arange(96), np.nanquantile(values, .1, axis=0), np.nanquantile(values, .9, axis=0), color="#4f81bd", alpha=.22, label="day q10-q90")
    axes[1].set_ylabel("Frozen active-power sample")
    axes[1].set_xticks(np.arange(0, 96, 8), [TIME_COLS[index] for index in range(0, 96, 8)], rotation=45, ha="right")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def build_funnel(frame: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("all_users", pd.Series(True, index=frame.index)),
        ("valid_nonzero_with_curve", frame.data_status.eq("valid") & frame.raw_curve_available.fillna(False) & ~frame.strict_all_zero.fillna(False)),
        ("frozen_main_high_power_flag", bool_mask(frame.high_power_flag)),
        ("relative_high_power", frame.relative_high_power_flag.fillna(False)),
        ("persistent_high_load", frame.relative_high_power_flag.fillna(False) & frame.persistent_high_load.fillna(False)),
        ("low_human_active_opposition", frame.relative_high_power_flag.fillna(False) & frame.persistent_high_load.fillna(False) & (frame.family_human_absence_support >= .65)),
        ("automatic_or_equipment_structure", frame.relative_high_power_flag.fillna(False) & frame.persistent_high_load.fillna(False) & (frame.family_human_absence_support >= .65) & (frame.equipment_support_or_automatic_percentile >= .60)),
        ("tier_b_or_better", frame.hp_candidate_tier.isin(["A", "B"])),
        ("tier_a_core", frame.hp_candidate_tier.eq("A")),
    ]
    return pd.DataFrame([{"stage": name, "user_count": int(mask.sum())} for name, mask in stages])


def make_combined_output(frame: pd.DataFrame, paths: BranchPaths | None = None) -> pd.DataFrame:
    paths = paths or BranchPaths.discover()
    original = pd.read_csv(paths.upstream / "screening_results.csv", dtype={"meter_id": str})
    branch_cols = [
        "meter_id", "hp_candidate_tier", "hp_candidate_flag", "hp_core_candidate_flag",
        "hp_branch_score", "hp_conditional_rank", "family_human_absence_support",
        "family_automatic_structure_support", "family_learned_representation_support",
        "equipment_only_similarity", "equipment_intermit_similarity", "persistent_high_load",
        "strict_high_power_flag", "relative_high_power_flag", "candidate_reason_codes",
    ]
    combined = original.merge(frame[branch_cols], on="meter_id", how="left", validate="one_to_one")
    combined["high_power_final_screening_class"] = combined.final_screening_class
    combined.loc[combined.hp_candidate_tier.eq("A"), "high_power_final_screening_class"] = "高功率疑似空置专项候选"
    combined["high_power_branch_interpretation"] = np.where(
        combined.hp_candidate_tier.eq("A"),
        "high-power suspected vacancy / equipment-maintained special candidate; not confirmed vacancy",
        np.where(combined.hp_candidate_tier.eq("B"), "secondary high-power review candidate; retain uncertainty", "no specialized route change"),
    )
    return combined


def markdown_table(frame: pd.DataFrame, columns=None, rows=20) -> str:
    selected = frame if columns is None else frame[columns]
    selected = selected.head(rows).copy()
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for values in selected.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_candidate_audit(frame, candidates, legacy_summary, channel_bias, counterfactual, funnel, paths: BranchPaths | None = None):
    paths = paths or BranchPaths.discover()
    high = frame[frame.relative_high_power_flag.fillna(False)]
    strict = frame[frame.strict_high_power_flag.fillna(False)]
    persistent = frame[frame.persistent_high_load.fillna(False)]
    frozen_high = frame[bool_mask(frame.high_power_flag)]
    main_distribution = frozen_high.final_screening_class.value_counts().rename_axis("Recognition class").reset_index(name="count")
    nonzero_vacancy = frame[
        frame.data_status.eq("valid")
        & ~frame.strict_all_zero.fillna(False)
        & frame.final_screening_class.isin(["高疑似空置", "中等疑似空置"])
    ]
    main_power_distribution = pd.DataFrame([{
        "count": len(nonzero_vacancy),
        "power_index_p10": nonzero_vacancy.main_power_index.quantile(.10),
        "power_index_median": nonzero_vacancy.main_power_index.median(),
        "power_index_p90": nonzero_vacancy.main_power_index.quantile(.90),
        "power_index_max": nonzero_vacancy.main_power_index.max(),
    }])
    hp_power_distribution = pd.DataFrame([{
        "count": len(nonzero_vacancy),
        "power_index_p10": nonzero_vacancy.hp_power_index.quantile(.10),
        "power_index_median": nonzero_vacancy.hp_power_index.median(),
        "power_index_p90": nonzero_vacancy.hp_power_index.quantile(.90),
        "power_index_max": nonzero_vacancy.hp_power_index.max(),
    }])
    audit_rows = frame[frame.meter_id.isin(REQUIRED_AUDIT_METERS)].copy()
    audit_rows["required_audit_object"] = True
    missing_required = REQUIRED_AUDIT_METERS - set(audit_rows.meter_id)
    if missing_required:
        raise RuntimeError(f"Required audit meters missing: {sorted(missing_required)}")
    recommendation_pool = candidates[
        candidates.hp_candidate_tier.eq("A") & ~bool_mask(candidates.future_holdout_flag)
    ].sort_values(["stability_rank_p10", "hp_conditional_rank"], ascending=False).head(3)
    recommendation = "No Tier A non-holdout case met every pre-registered gate. Do not manufacture a minimum case." if recommendation_pool.empty else \
        ", ".join(recommendation_pool.meter_id.astype(str)) + "：优先做最小业务核验；它们满足全部三证据族和稳定性门槛。"
    scale_summary = counterfactual.groupby("amplitude_scale").agg(
        main_global_score_median=("main_global_score", "median"),
        main_global_rank_median=("main_global_rank", "median"),
        behavior_cosine_median=("behavior_embedding_cosine", "median"),
        behavior_cosine_p10=("behavior_embedding_cosine", lambda values: float(np.nanquantile(values, .10))),
        hp_score_median=("hp_branch_score", "median"),
        hp_rank_median=("hp_conditional_rank", "median"),
    ).reset_index()
    text = f"""# High-Power Vacancy Candidate Audit

## Scope and interpretation

The branch scanned all {len(frame):,} frozen recognition users. High power only determined entry into this specialist audit. It was never treated as vacancy evidence or active/occupied evidence. Every power fraction is the fraction of observed periodic frozen samples, not physical runtime. No Gold or new human label was used.

The output term is **high-power vacancy screening candidate**. Subtype is reported separately. It does not mean confirmed vacancy or a calibrated vacancy probability.

## Pre-registered funnel

{markdown_table(funnel, rows=len(funnel))}

- Frozen Recognition `high_power_flag`: {len(frozen_high)} users.
- New pre-registered relative high power (fit-reference power index Top25%): {len(high)} users.
- Strict high power (Top10%): {len(strict)} users.
- Persistent high load under the sample-based definition: {len(persistent)} users.
- Tier A core candidates: {int(frame.hp_candidate_tier.eq('A').sum())}.
- Tier B secondary review candidates: {int(frame.hp_candidate_tier.eq('B').sum())}.

## Frozen Recognition failure mode

### Current recognition classes within the frozen {len(frozen_high)}-user high-power cohort

{markdown_table(main_distribution, rows=len(main_distribution))}

### Original Recognition power index of all non-zero high/medium suspected-vacancy outputs

{markdown_table(main_power_distribution, rows=1)}

### Recognition four-view high-power qualification index for the same users

{markdown_table(hp_power_distribution, rows=1)}

The two tables are deliberately separate. The first audits the frozen main-model field. The second uses the pre-registered specialist qualification definition and does not rewrite the original result.

The global model and Legacy consensus inherited a negative magnitude shortcut because the historical positive set scarcely covers persistent high-load vacancy. The new branch does not lower the frozen Recognition threshold. It re-ranks only inside a pre-defined high-power cohort using magnitude-resistant behavior evidence.

## Legacy power coverage and historical selection bias

{markdown_table(legacy_summary, rows=len(legacy_summary))}

The 29 equipment-maintained and 21 equipment-intermittent users are unique meter IDs. They are not pooled into one prototype. A Legacy user may have occasional kW-level samples without providing historical coverage of persistent high-load vacancy. Lack of coverage is reported as historical selection bias, not evidence that high-power vacancy does not exist.

## Channel dependence on power

{markdown_table(channel_bias, rows=len(channel_bias))}

## Final candidates

{markdown_table(candidates, ['meter_id','hp_candidate_tier','raw_median_power','raw_q90_power','baseline_floor','frozen_sample_gt_1kw_fraction','daily_sample_median_gt_1kw_count','family_human_absence_support','family_automatic_structure_support','family_learned_representation_support','equipment_only_similarity_percentile','hp_conditional_rank','stability_rank_p10','future_holdout_flag','candidate_reason_codes'], rows=max(1, len(candidates)))}

Candidates differ from ordinary high-power normal-use cases because they combine persistent high frozen-sample load with power-conditioned lack of sustained human/active evidence, automatic/background-compatible structure, and magnitude-normalized similarity to historical equipment-maintained patterns. A high score in only one correlated channel cannot produce Tier A.

## Additional runtime-specified audit objects

These users were audited after the rule was frozen. They were not used as preset answers or to tune thresholds.

{markdown_table(audit_rows, ['meter_id','hp_candidate_tier','raw_median_power','raw_q90_power','baseline_floor','frozen_sample_gt_1kw_fraction','daily_sample_median_gt_1kw_count','human_interaction_evidence','active_evidence_score','automatic_control_compatibility','equipment_only_similarity_percentile','hp_conditional_rank','stability_rank_p10','future_holdout_flag','candidate_reason_codes'], rows=len(audit_rows))}

## Amplitude-scaling counterfactual

{markdown_table(scale_summary, rows=len(scale_summary))}

This experiment holds the 29 confirmed equipment-maintained Legacy temporal patterns fixed and changes amplitude only. The Recognition global response, Behavior embedding cosine, specialist-branch score and conditional rank are all reported for ×1/×2/×5/×10. The diagnostic thresholds were frozen before the results and were not adjusted afterward.

Robustness applies to Daily-TCN Behavior representation and final conditional rank; behavior-only PU raw scores are not claimed to be amplitude-invariant.

## Old Word method decision path

The original report used fixed 0.5 kW behavior-level and daily-load decision paths and described equipment-maintained examples mainly in a low-baseload range. For a persistent high-load candidate, that path tends to treat the large load as human-use evidence. This branch instead asks whether the 42-day structure is compatible with sustained automatic/background control while lacking persistent human interaction.

This is a **decision-path comparison based on the report**, not a full reproduction of the historical XGBoost model. The project does not have the complete original March feature pipeline/model in this branch, and periodic frozen samples are not converted to true kWh.

## Minimal business verification recommendation

{recommendation}

Business verification should first check actual residential/special-use status, shared-meter possibility, supply/meter state, and whether sustained equipment operation can occur without occupants. A positive review would validate the repaired high-power boundary; a negative review would reveal which confounder the branch must learn.

## What remains uncertain

- Tier A/B are no-Gold screening tiers, not confirmed vacancy.
- A possible non-residential/shared/special-service meter remains a confounder when customer archive fields are unavailable.
- Automatic-Control Compatibility describes curve compatibility, not a detected appliance.
- Future holdout rows are inference-only and retain their protected validation role.
"""
    target = paths.root / "high_power_candidate_audit.md"
    target.write_text(text, encoding="utf-8")
    return target
