from __future__ import annotations

import json
import pandas as pd
from .config import SEEDS, WINDOWS, BranchPaths
from .model import fit_high_power_branch


def run_branch(audit_amplitude=False):
    """Fit the published branch only inside a separate private workspace."""
    from .config import decision_freeze_payload, write_decision_freeze
    from .counterfactual import run_amplitude_counterfactual
    from .data import build_power_audit, load_unique_legacy, build_legacy_power_coverage, channel_power_bias
    from .model import assign_tiers, save_branch_model
    from .stability import build_stability_from_variants
    from .reporting import add_confounders, build_funnel, make_combined_output, verify_combined_output, candidate_columns, plot_meter, write_candidate_audit

    paths = BranchPaths.discover()
    if (paths.root / "models" / "high_power_branch_model.joblib").exists():
        raise RuntimeError("Branch model already exists and is treated as frozen; choose another workspace")
    paths.root.mkdir(parents=True, exist_ok=True)
    write_decision_freeze(paths)
    frozen_decision = decision_freeze_payload(paths)
    power_frame, _ = build_power_audit(paths)
    legacy, _ = load_unique_legacy(paths)
    main_frame, artifact, audit = fit_high_power_branch(power_frame, paths)
    save_branch_model(artifact, audit, paths)
    scenarios = stability_scenarios(power_frame, paths)
    long, stability = build_stability_from_variants(scenarios, set(artifact["fit_reference_ids"]))
    main_frame = main_frame.drop(columns=[column for column in main_frame if column.startswith("stability_")], errors="ignore")
    main_frame = main_frame.merge(stability, on="meter_id", how="left", validate="one_to_one")
    main_frame = add_confounders(assign_tiers(main_frame))
    high = main_frame.loc[main_frame.relative_high_power_flag | main_frame.strict_high_power_flag | main_frame.persistent_high_load]
    candidates = main_frame.loc[main_frame.hp_candidate_tier.isin(["A", "B"])].sort_values(
        ["hp_candidate_tier", "stability_rank_p10", "hp_conditional_rank"], ascending=[True, False, False]
    )
    for frame, filename in [
        (high, "all_high_power_users.csv"),
        (candidates[[column for column in candidate_columns() if column in candidates]], "high_power_vacancy_candidates.csv"),
        (stability, "high_power_candidate_stability.csv"),
        (long.loc[long.meter_id.isin(high.meter_id)], "high_power_stability_long.csv"),
        (build_funnel(main_frame), "high_power_funnel.csv"),
    ]:
        frame.to_csv(paths.root / filename, index=False, encoding="utf-8-sig")
    combined = make_combined_output(main_frame, paths)
    combined_path = paths.root / "screening_results_with_high_power.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    verify_combined_output(paths, combined_path)
    if audit_amplitude:
        counterfactual = run_amplitude_counterfactual(main_frame, artifact, paths)
        counterfactual.to_csv(paths.root / "amplitude_scaling_counterfactual.csv", index=False)
        write_candidate_audit(main_frame, candidates, build_legacy_power_coverage(power_frame, legacy),
                              channel_power_bias(power_frame), counterfactual, build_funnel(main_frame), paths)
    plot_dir = paths.root / "case_plots"
    plot_dir.mkdir(exist_ok=True)
    daily = load_daily_for_plots(paths)
    for index, row in enumerate(candidates.head(10).itertuples(index=False), 1):
        plot_meter(daily.loc[daily.meter_id.eq(str(row.meter_id))], plot_dir / f"candidate_{index:02d}.png",
                   f"{row.meter_id} | Tier {row.hp_candidate_tier} | high-power suspected-vacancy specialist candidate")
    if frozen_decision != decision_freeze_payload(paths):
        raise RuntimeError("Decision specification changed during execution")
    print(json.dumps({"users": len(main_frame), "tier_a": int(candidates.hp_candidate_tier.eq('A').sum()),
                      "tier_b": int(candidates.hp_candidate_tier.eq('B').sum()), "holdout_fit_overlap": audit['future_holdout_overlap_count']}, indent=2))


def stability_scenarios(power_frame, paths):
    scenarios = []
    seen = set()
    for seed in SEEDS:
        for window in WINDOWS:
            seen.add((seed, window, .10, .50))
    for prior in (.06, .16):
        seen.add((SEEDS[0], 42, prior, .50))
    for weight in (.25, .75):
        seen.add((SEEDS[0], 42, .10, weight))
    for index, (seed, window, prior, weight) in enumerate(sorted(seen)):
        print(json.dumps({"stability_fit": index + 1, "total": len(seen), "seed": seed, "window": window, "prior": prior, "medium_weight": weight}), flush=True)
        frame, _, _ = fit_high_power_branch(
            power_frame, paths, seed=seed, embedding_seed=seed, window=window,
            prior=prior, medium_weight=weight, bags=3,
        )
        frame.attrs.update({"seed": seed, "window_days": window, "prior": prior, "medium_weight": weight})
        scenarios.append(frame)
    return scenarios


def load_daily_for_plots(paths):
    daily = pd.read_pickle(paths.upstream / "inputs" / "data_snapshot" / "daily_curves.pkl").copy().reset_index(drop=True)
    daily["meter_id"] = daily.meter_id.astype(str).str.replace(r"\.0$", "", regex=True)
    daily["date"] = pd.to_datetime(daily.date)
    return daily
