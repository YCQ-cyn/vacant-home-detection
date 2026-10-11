"""Post-freeze regional/hardware matched-peer audit with no learning step."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = (
    "human_interaction_evidence", "active_evidence_score", "automatic_control_compatibility",
    "family_human_absence_support", "family_automatic_structure_support",
    "family_learned_representation_support", "hp_conditional_rank",
)
MATCHING_SPEC = {
    "exact_terminal_minimum": 10,
    "nearest_peer_limit": 30,
    "distance_field": "hp_power_index",
    "matching_fields": ["utility", "meter_vendor", "meter_version", "terminal_vendor"],
    "percentile": "midrank empirical CDF",
    "exclude_target": True,
    "exclude_entire_future_holdout": True,
    "audit_only": True,
}


def flags(values):
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def midrank_percentile(value, peers):
    peers = pd.to_numeric(pd.Series(peers), errors="coerce").dropna().to_numpy(float)
    if not np.isfinite(value) or not len(peers):
        return np.nan
    equal = np.isclose(peers, value, rtol=1e-12, atol=1e-12)
    below = (peers < value) & ~equal
    return float((below.sum() + .5 * equal.sum()) / len(peers))


def match_peers(frame, target_id, protected_ids):
    target = frame.loc[frame.meter_id.eq(target_id)]
    if len(target) != 1:
        raise ValueError("Each requested target must occur exactly once")
    target = target.iloc[0]
    pool = frame.loc[
        ~frame.meter_id.isin(set(protected_ids) | {target_id})
        & ~flags(frame.data_invalid)
        & ~flags(frame.strict_all_zero)
        & pd.to_numeric(frame.hp_power_index, errors="coerce").notna()
    ].copy()
    for column in ("utility", "meter_vendor", "meter_version"):
        if pd.isna(target[column]):
            return pool.iloc[0:0], "MISSING_MATCH_METADATA"
        pool = pool.loc[pool[column].eq(target[column])]
    exact = pool.loc[pool.terminal_vendor.eq(target.terminal_vendor)] if pd.notna(target.terminal_vendor) else pool.iloc[0:0]
    level = "L1_EXACT_TERMINAL" if len(exact) >= MATCHING_SPEC["exact_terminal_minimum"] else "L2_HARDWARE_REGION"
    selected = (exact if level == "L1_EXACT_TERMINAL" else pool).copy()
    selected["power_index_abs_distance"] = (
        pd.to_numeric(selected.hp_power_index, errors="coerce") - float(target.hp_power_index)
    ).abs()
    return selected.sort_values(["power_index_abs_distance", "meter_id"]).head(MATCHING_SPEC["nearest_peer_limit"]), level


def audit_targets(frame, target_ids, protected_ids):
    if frame.meter_id.duplicated().any():
        raise ValueError("Duplicate meter identifiers in audit frame")
    rows, peer_frames = [], []
    for target_id in target_ids:
        target = frame.loc[frame.meter_id.eq(target_id)].iloc[0]
        peers, level = match_peers(frame, target_id, protected_ids)
        row = {"meter_id": target_id, "matching_level": level, "peer_count": len(peers)}
        for column in ("utility", "meter_vendor", "meter_version", "terminal_vendor", "hp_power_index"):
            row[column] = target[column]
        for column in METRICS:
            row[column] = target[column]
            row[f"{column}_local_percentile"] = midrank_percentile(target[column], peers[column])
        row["power_distance_median"] = peers.power_index_abs_distance.median() if len(peers) else np.nan
        row["power_distance_max"] = peers.power_index_abs_distance.max() if len(peers) else np.nan
        row["terminal_vendor_match_rate"] = float(peers.terminal_vendor.eq(target.terminal_vendor).mean()) if len(peers) else np.nan
        rows.append(row)
        peer_frames.append(peers.assign(audit_target=target_id))
    return pd.DataFrame(rows), pd.concat(peer_frames, ignore_index=True) if peer_frames else pd.DataFrame()


def run_audit(scores_path, hardware_path, holdout_path, target_ids, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.dumps(MATCHING_SPEC, ensure_ascii=False, sort_keys=True, indent=2)
    frozen = output_dir / "matching_spec.json"
    if frozen.exists() and frozen.read_text(encoding="utf-8") != spec:
        raise RuntimeError("Existing matching specification differs; use a separate audit directory")
    frozen.write_text(spec, encoding="utf-8")
    spec_hash = hashlib.sha256(spec.encode("utf-8")).hexdigest()
    scores = pd.read_csv(scores_path, dtype={"meter_id": str})
    hardware = pd.read_csv(hardware_path, dtype={"meter_id": str})
    held = set(pd.read_csv(holdout_path, dtype={"meter_id": str}).meter_id)
    hardware_columns = ["meter_id", "utility", "meter_vendor", "meter_version", "terminal_vendor"]
    frame = scores.drop(columns=[column for column in hardware_columns[1:] if column in scores]).merge(
        hardware[hardware_columns], on="meter_id", how="left", validate="one_to_one"
    )
    audit, peers = audit_targets(frame, target_ids, held)
    if set(peers.meter_id) & held:
        raise RuntimeError("Protected holdout entered matched-peer reference")
    audit.to_csv(output_dir / "matched_peer_audit.csv", index=False, encoding="utf-8-sig")
    peers.to_csv(output_dir / "matched_peers.csv", index=False, encoding="utf-8-sig")
    (output_dir / "audit_integrity.json").write_text(json.dumps({
        "spec_sha256": spec_hash, "target_count": len(audit), "holdout_peer_overlap": 0,
        "model_fitting_performed": False, "candidate_tiers_modified": False,
    }, indent=2), encoding="utf-8")
    print(f"Matched-peer audit complete: {len(audit)} targets; holdout reference overlap: 0")

