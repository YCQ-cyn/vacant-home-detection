from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import ROOT, TIME_COLS, load_daily, write_json


def write_local_report():
    """Generate local case figures and an evidence summary without exporting data."""
    import json

    results = pd.read_csv(ROOT / "screening_results.csv", dtype={"meter_id": str})
    results = add_output_contract(results)
    (ROOT / "case_plots").mkdir(parents=True, exist_ok=True)
    top = select_top_cases(results)
    daily = load_daily()
    for index, row in enumerate(top.itertuples(index=False), 1):
        plot_case(row.meter_id, row.case_category, daily, ROOT / "case_plots" / f"case_{index:02d}.png")
    selection = json.loads((ROOT / "model_selection.json").read_text(encoding="utf-8"))
    counts = results.final_screening_class.value_counts().to_dict()
    summary = {
        "selected_structure": selection["selected_family"],
        "user_count": int(len(results)),
        "screening_class_counts": counts,
        "evaluation_qualification": "PU labels held out per meter; SSL research representations are transductive over non-holdout research users",
        "calibrated_vacancy_probability": False,
        "fresh_negative_gold_available": False,
        "independent_automatic_human_axes": True,
    }
    write_json(ROOT / "recognition_summary.json", summary)
    body = "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n"
    (ROOT / "recognition_report.md").write_text(
        "# 空置房识别证据报告\n\n中文 | [English](recognition_report.en.md)\n\n"
        "分数为证据排名；不代表真实空置概率或经Gold验证的准确率。\n\n" + body,
        encoding="utf-8",
    )
    (ROOT / "recognition_report.en.md").write_text(
        "# Vacant-home Recognition Evidence Report\n\n[中文](recognition_report.md) | English\n\n"
        "Scores are evidence ranks, not calibrated vacancy probabilities or Gold-validated accuracy.\n\n" + body,
        encoding="utf-8",
    )


def bool_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_output_contract(results):
    results = results.copy()
    results["vacancy_screening_rank"] = results.vacancy_screening_score
    required = [
        "meter_id", "data_status", "silence_status", "vacancy_screening_score", "vacancy_screening_rank",
        "final_screening_class", "automatic_control_compatibility", "human_interaction_evidence",
        "model_conflict", "model_disagreement", "complete_support", "equipment_support",
        "equipment_intermit_support", "intermittent_support", "subtype_entropy", "subtype_margin",
        "weekend_tendency", "high_power_flag", "future_holdout_flag", "reason_codes",
    ]
    remaining = [column for column in results if column not in required]
    results = results[required + remaining]
    results.to_csv(ROOT / "screening_results.csv", index=False, encoding="utf-8-sig")
    return results


def select_top_cases(results):
    chosen, rows = set(), []

    def take(category, frame, n=3):
        for _, row in frame.iterrows():
            if row.meter_id in chosen:
                continue
            chosen.add(row.meter_id)
            rows.append({"meter_id": row.meter_id, "case_category": category})
            if sum(item["case_category"] == category for item in rows) >= n:
                break

    nonzero = results[
        ~results.silence_status.eq("valid_strict_all_zero")
        & results.data_status.eq("valid")
        & ~results.future_holdout_flag.fillna(False).astype(bool)
    ]
    take("novel_high_vacancy", nonzero.sort_values(["vacancy_screening_score", "legacy_anchor_rank"], ascending=[False, True]), 4)
    take("high_power_automatic_candidate", nonzero[bool_mask(nonzero.high_power_automatic_challenge)].sort_values("vacancy_screening_score", ascending=False), 4)
    take("low_power_rich_activity", nonzero[bool_mask(nonzero.low_power_rich_activity_challenge)].sort_values("active_evidence_score", ascending=False), 3)
    take("vacancy_active_conflict", nonzero[bool_mask(nonzero.model_conflict)].sort_values("model_disagreement", ascending=False), 3)
    take("uncertain_disagreement", nonzero.sort_values("model_disagreement", ascending=False), 3)
    for subtype in ["complete_support", "equipment_support", "equipment_intermit_support", "intermittent_support"]:
        take(f"subtype_{subtype.removesuffix('_support')}", nonzero.sort_values(subtype, ascending=False), 1)
    output = pd.DataFrame(rows).merge(results, on="meter_id", how="left")
    output.to_csv(ROOT / "case_plots" / "top_case_index.csv", index=False, encoding="utf-8-sig")
    return output


def plot_case(meter_id, category, daily, output_path, title_suffix=""):
    group = daily[daily.meter_id.eq(str(meter_id))].sort_values("date")
    if group.empty:
        return False
    values = group[TIME_COLS].to_numpy(float)
    transformed = np.sign(values) * np.log1p(np.abs(values))
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    image = axes[0].imshow(transformed, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title(f"{meter_id} | {category}{title_suffix}")
    axes[0].set_ylabel("Day (chronological)")
    axes[0].set_xticks(np.arange(0, 96, 8), [TIME_COLS[index] for index in range(0, 96, 8)], rotation=45, ha="right")
    figure.colorbar(image, ax=axes[0], label="signed log(1 + sampled power)")
    x = np.arange(96)
    axes[1].plot(x, np.nanmedian(values, axis=0), label="median day", color="#1f4e79", linewidth=1.8)
    axes[1].fill_between(x, np.nanquantile(values, .1, axis=0), np.nanquantile(values, .9, axis=0), alpha=.22, color="#4f81bd", label="day q10-q90")
    axes[1].set_xticks(np.arange(0, 96, 8), [TIME_COLS[index] for index in range(0, 96, 8)], rotation=45, ha="right")
    axes[1].set_ylabel("Periodic frozen power sample")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True
