from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import BranchPaths, SEEDS
from .data import empirical_cdf, load_unique_legacy
from .model import _cosine_rows


def load_frozen_global_artifact(paths: BranchPaths | None = None):
    paths = paths or BranchPaths.discover()
    return joblib.load(paths.upstream / "models" / "final_pu_channels.joblib")


def counterfactual_reference_similarity(vector, artifact: dict, meter_id: str, kind: str) -> float:
    if kind == "equipment":
        ids = artifact["equipment_reference_ids"]
        references = artifact["equipment_reference_vectors"]
    else:
        ids = artifact["equipment_intermittent_reference_ids"]
        references = artifact["equipment_intermittent_reference_vectors"]
    exclude = ids.index(str(meter_id)) if str(meter_id) in ids else None
    return _cosine_rows(vector, references, exclude_index=exclude)


def scaled_daily_inputs(x: np.ndarray, scale: float) -> np.ndarray:
    result = x.copy()
    clean = np.sign(x[:, 0]) * np.expm1(np.abs(x[:, 0])) * float(scale)
    signed_log = np.sign(clean) * np.log1p(np.abs(clean))
    median = np.nanmedian(clean, axis=1, keepdims=True)
    q25 = np.nanquantile(clean, .25, axis=1, keepdims=True)
    q75 = np.nanquantile(clean, .75, axis=1, keepdims=True)
    shape = np.clip((clean - median) / np.maximum(q75 - q25, .02), -12, 12)
    difference = np.diff(signed_log, axis=1, prepend=signed_log[:, :1])
    relative_difference = np.diff(clean, axis=1, prepend=clean[:, :1]) / np.maximum(np.abs(clean), .05)
    result[:, 0] = signed_log
    result[:, 1] = shape
    result[:, 2] = difference
    result[:, 3] = np.clip(relative_difference, -10, 10)
    return result.astype(np.float32)


def _load_daily_models(paths: BranchPaths):
    from ..representation import DailyDualTCN
    models = {}
    for seed in SEEDS:
        checkpoint = torch.load(
            paths.upstream / "models" / "representations" / "final_plain" / f"daily_seed_{seed}" / "daily_encoder.pt",
            map_location="cpu", weights_only=False,
        )
        model = DailyDualTCN(int(checkpoint["hidden"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[seed] = model
    return models


def _embedding_response(models, x):
    full, behavior = [], []
    tensor = torch.from_numpy(x)
    with torch.no_grad():
        for seed in SEEDS:
            z, b, *_ = models[seed](tensor)
            full.append(z.mean(0).numpy())
            behavior.append(b.mean(0).numpy())
    return np.mean(full, axis=0), np.mean(behavior, axis=0)


def _synthetic_feature_row(original: pd.Series, full_embedding: np.ndarray, scale: float) -> pd.DataFrame:
    row = original.copy()
    linear_columns = [
        "raw_q10_power", "raw_median_power", "raw_q90_power", "baseline_floor",
        "clock_time_recurrence_dispersion", "change_point_structure",
        "raw_early_late_shape_distance", "raw_weekday_weekend_shape_difference",
        "weekend_q90_difference",
    ]
    for column in linear_columns:
        if column in row and pd.notna(row[column]):
            row[column] = float(row[column]) * scale
    for index in range(64):
        row[f"daily_mean_e{index:02d}"] = float(full_embedding[index])
    return pd.DataFrame([row])


def _scaled_power_condition_metrics(x: np.ndarray) -> dict:
    raw = np.sign(x[:, 0]) * np.expm1(np.abs(x[:, 0]))
    finite = raw[np.isfinite(raw)]
    daily_q10 = np.nanquantile(raw, .10, axis=1)
    return {
        "raw_median_power": float(np.nanmedian(finite)),
        "raw_q90_power": float(np.nanquantile(finite, .90)),
        "baseline_floor": float(np.nanmedian(daily_q10)),
        "frozen_sample_gt_1kw_fraction": float(np.mean(finite > 1.0)),
    }


def run_amplitude_counterfactual(main_frame: pd.DataFrame, hp_artifact: dict, paths: BranchPaths | None = None) -> pd.DataFrame:
    paths = paths or BranchPaths.discover()
    legacy, _ = load_unique_legacy(paths)
    equipment_ids = legacy.loc[legacy.subtype.eq("equipment"), "meter_id"].tolist()
    x = np.load(paths.upstream / "inputs" / "daily_inputs.npz")["x"].astype(np.float32)
    index = pd.read_csv(paths.upstream / "inputs" / "daily_input_index.csv", dtype={"meter_id": str})
    index["meter_id"] = index.meter_id.astype(str)
    feature_table = pd.read_parquet(paths.upstream / "final_fit_feature_table.parquet")
    feature_table["meter_id"] = feature_table.meter_id.astype(str)
    feature_lookup = feature_table.set_index("meter_id", drop=False)
    global_artifact = load_frozen_global_artifact(paths)["selected_pu_rank"]
    global_models = global_artifact["models"]
    all_global = np.mean([model.predict_proba(feature_table) for model in global_models], axis=0)
    global_fit = feature_table.meter_id.isin(set(global_artifact["fit_meter_ids"])).to_numpy()
    global_reference = np.sort(all_global[global_fit])
    daily_models = _load_daily_models(paths)
    base_by_meter = main_frame.set_index("meter_id")
    response_cache = {}
    rows = []
    for meter_id in equipment_ids:
        positions = index.index[index.meter_id.eq(meter_id)].to_numpy()
        original_feature = feature_lookup.loc[meter_id]
        original_hp = base_by_meter.loc[meter_id]
        for scale in (1, 2, 5, 10):
            scaled = scaled_daily_inputs(x[positions], scale)
            full_embedding, behavior_embedding = _embedding_response(daily_models, scaled)
            response_cache[(meter_id, scale)] = behavior_embedding
            synthetic = _synthetic_feature_row(original_feature, full_embedding, scale)
            condition_metrics = _scaled_power_condition_metrics(scaled)
            for column, value in condition_metrics.items():
                synthetic[column] = value
            global_score = float(np.mean([model.predict_proba(synthetic)[0] for model in global_models]))
            global_rank = float(np.searchsorted(global_reference, global_score, side="right") / len(global_reference))
            hp_synthetic = synthetic.copy()
            for index_value in range(32):
                hp_synthetic[f"behavior_mean_e{index_value:02d}"] = float(behavior_embedding[index_value])
            pu_score = float(hp_artifact["behavior_pu"].predict(hp_synthetic)[0])
            equipment_similarity = counterfactual_reference_similarity(behavior_embedding, hp_artifact, meter_id, "equipment")
            intermit_similarity = counterfactual_reference_similarity(behavior_embedding, hp_artifact, meter_id, "equipment_intermittent")
            distributions = hp_artifact["high_power_reference_distributions"]
            equipment_pct = float(empirical_cdf([equipment_similarity], distributions["equipment_only_similarity"])[0])
            intermit_pct = float(empirical_cdf([intermit_similarity], distributions["equipment_intermit_similarity"])[0])
            pu_pct = float(empirical_cdf([pu_score], distributions["behavior_only_pu_score"])[0])
            representation_support = float(np.nanmedian([equipment_pct, intermit_pct, pu_pct]))
            human_value = float(original_hp.human_interaction_evidence)
            active_value = float(original_hp.active_evidence_score)
            human_pct = float(hp_artifact["conditioner"].conditional_percentile(
                synthetic, np.array([human_value]), hp_artifact["conditioner_reference_human"]
            )[0])
            active_pct = float(hp_artifact["conditioner"].conditional_percentile(
                synthetic, np.array([active_value]), hp_artifact["conditioner_reference_active"]
            )[0])
            human_absence_support = float(np.nanmedian([1 - human_pct, 1 - active_pct]))
            hp_score = float(np.nanmedian([
                human_absence_support,
                original_hp.family_automatic_structure_support,
                representation_support,
            ]))
            hp_rank = float(empirical_cdf([hp_score], hp_artifact["high_power_reference_scores"])[0])
            rows.append({
                "meter_id": meter_id,
                "legacy_subtype": "equipment",
                "amplitude_scale": scale,
                "main_global_score": global_score,
                "main_global_rank": global_rank,
                "behavior_embedding_cosine": np.nan,
                "power_conditioned_human_absence_support": human_absence_support,
                "automatic_structure_support_held_by_design": float(original_hp.family_automatic_structure_support),
                "hp_branch_score": hp_score,
                "hp_conditional_rank": hp_rank,
                "equipment_only_similarity": equipment_similarity,
                "equipment_intermit_similarity": intermit_similarity,
                "behavior_only_pu_score": pu_score,
                "point_semantics": "15-minute periodic frozen active-power samples",
            })
    output = pd.DataFrame(rows)
    for meter_id in equipment_ids:
        baseline = response_cache[(meter_id, 1)]
        denominator = np.linalg.norm(baseline)
        for scale in (1, 2, 5, 10):
            current = response_cache[(meter_id, scale)]
            cosine = float(baseline @ current / (denominator * np.linalg.norm(current) + 1e-12))
            output.loc[(output.meter_id.eq(meter_id)) & output.amplitude_scale.eq(scale), "behavior_embedding_cosine"] = cosine
    return output
