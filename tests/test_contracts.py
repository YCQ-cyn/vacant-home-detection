"""Synthetic tests: no household data, labels or fitted private artifacts."""
from __future__ import annotations

import importlib
import json
import pkgutil

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from vacant_home import cli
from vacant_home.high_power.config import DECISION
from vacant_home.high_power.counterfactual import counterfactual_reference_similarity, scaled_daily_inputs
from vacant_home.high_power.data import empirical_cdf, sample_power_metrics, TIME_COLS
from vacant_home.high_power.model import PowerConditioner, assign_tiers
from vacant_home.inference import score_feature_table
from vacant_home.models import MixedEstimator
from vacant_home.peer_audit import METRICS, audit_targets, match_peers, midrank_percentile
from vacant_home.representation import DailyDualTCN, LongEncoder, balanced_indices, mask_blocks


def test_all_modules_import_without_private_inputs():
    import vacant_home
    for entry in pkgutil.walk_packages(vacant_home.__path__, vacant_home.__name__ + "."):
        if not entry.name.endswith(".__main__"):
            importlib.import_module(entry.name)


def test_status_does_not_create_workspace(tmp_path, capsys):
    private = tmp_path / "private"
    cli.main(["--workspace", str(private), "status"])
    status = json.loads(capsys.readouterr().out)
    assert not any(status["ready"].values())
    assert not private.exists()


@pytest.mark.parametrize("hidden", [32, 64])
def test_daily_encoder_shapes_and_masking(hidden):
    torch.manual_seed(11)
    data = torch.randn(3, 8, 96)
    outputs = DailyDualTCN(hidden).eval()(data)
    assert [tuple(x.shape) for x in outputs] == [(3, 64), (3, 32), (3, 32), (3, 64, 96), (3, 96), (3, 3)]
    assert all(torch.isfinite(x).all() for x in outputs)
    view, mask = mask_blocks(data, torch.Generator().manual_seed(17))
    assert ((mask.sum(1) >= 8) & (mask.sum(1) <= 20)).all()
    assert torch.all(view[:, 4][mask] == 1)
    assert torch.equal(data[:, 0][~mask], view[:, 0][~mask])


@pytest.mark.parametrize("kind", ["tcn", "gru"])
def test_long_encoder_windows(kind):
    model = LongEncoder(kind).eval()
    for width in (14, 21, 28, 42):
        sequence = torch.randn(2, width, 64)
        outputs = model(sequence, torch.ones(2, width))
        assert outputs[0].shape == (2, 64)
        assert outputs[2].shape == (2, width, 64)
        assert all(torch.isfinite(x).all() for x in outputs)


def test_balanced_sampler_keeps_protected_users_out():
    index = pd.DataFrame({"meter_id": ["active"] * 100 + ["zero"] * 100 + ["held"] * 42})
    selected = balanced_indices(index, {"active", "zero"}, {"zero"}, .10, 11)
    meters = index.iloc[selected].meter_id
    assert "held" not in set(meters)
    assert (meters == "zero").mean() <= .10
    assert len(set(selected)) == len(selected)


def candidate_frame():
    return pd.DataFrame({
        "data_status": ["valid"] * 4,
        "relative_high_power_flag": [True] * 4,
        "strict_high_power_flag": [True] * 4,
        "persistent_high_load": [True, True, False, True],
        "severe_quality_issue": [False, False, False, True],
        "occasional_high_peak_only": [False, False, True, False],
        "family_human_absence_support": [.95] * 4,
        "family_automatic_structure_support": [.95] * 4,
        "family_learned_representation_support": [.95, np.nan, .95, .95],
        "equipment_only_similarity_percentile": [.95] * 4,
        "equipment_support_or_automatic_percentile": [.95] * 4,
        "stability_rank_median": [.95] * 4,
        "stability_rank_p10": [.90] * 4,
    })


def test_tier_requires_three_valid_families_and_persistence():
    output = assign_tiers(candidate_frame())
    assert output.hp_candidate_tier.tolist() == ["A", "B", "C", "C"]
    assert output.hp_core_candidate_flag.tolist() == [True, False, False, False]


def test_frozen_tier_gate_boundary():
    frame = candidate_frame().iloc[:1].copy()
    frame["stability_rank_p10"] = DECISION["tier_a_rank_p10"] - .001
    assert assign_tiers(frame).hp_candidate_tier.iloc[0] != "A"


@pytest.mark.parametrize("scale", [1, 2, 5, 10])
def test_amplitude_counterfactual_retains_masks_and_clock(scale):
    raw = np.linspace(.05, 2, 192).reshape(2, 96)
    x = np.zeros((2, 8, 96), np.float32)
    x[:, 0] = np.log1p(raw)
    x[:, 4:, 12:14] = 1
    result = scaled_daily_inputs(x, scale)
    np.testing.assert_allclose(np.expm1(result[:, 0]), raw * scale, rtol=1e-6)
    np.testing.assert_array_equal(result[:, 4:], x[:, 4:])
    assert np.argmax(result[0, 0]) == np.argmax(raw[0])


def test_similarity_excludes_own_legacy_reference():
    references = np.array([[1., 0.], [0., 1.]])
    artifact = {"equipment_reference_ids": ["toy_a", "toy_b"], "equipment_reference_vectors": references}
    assert counterfactual_reference_similarity(references[0], artifact, "toy_a", "equipment") == 0


def test_power_conditioner_uses_fixed_reference():
    reference = pd.DataFrame({"meter_id": ["fit_a", "fit_b", "fit_c"], "scale": [1., 2., 3.]})
    model = PowerConditioner(["scale"], neighbors=3).fit(reference)
    held = pd.DataFrame({"meter_id": ["held"], "scale": [2.]})
    assert "held" not in model.reference_meter_ids
    np.testing.assert_allclose(model.conditional_percentile(held, [.5], [.1, .5, .9]), [2 / 3])
    np.testing.assert_allclose(empirical_cdf([.5, np.nan], [.1, .5, .9]), [2 / 3, np.nan])


def test_persistence_is_sample_based_and_peak_is_not_baseload():
    values = np.full((42, 96), 2.)
    values[0, 0] = 8
    curves = pd.DataFrame(values, columns=TIME_COLS)
    curves["meter_id"] = "persistent"
    curves["date"] = pd.date_range("2024-07-01", periods=42)
    curves["unresolved_missing_points"] = 0
    curves["original_missing_points"] = 0
    row = sample_power_metrics(curves).iloc[0]
    assert row.raw_median_power == 2
    assert row.frozen_sample_gt_1kw_fraction == 1
    assert row.daily_sample_median_gt_1kw_count == 42


def test_peer_matching_excludes_holdout_and_target():
    frame = pd.DataFrame({
        "meter_id": ["target", "held", "peer", "other_region"],
        "utility": ["area_a", "area_a", "area_a", "area_b"],
        "meter_vendor": ["vendor"] * 4, "meter_version": ["spec"] * 4,
        "terminal_vendor": ["terminal"] * 4,
        "data_invalid": [False] * 4, "strict_all_zero": [False] * 4,
        "hp_power_index": [.8, .8, .81, .8],
    })
    for column in METRICS:
        frame[column] = [.9, .99, .5, .1]
    peers, level = match_peers(frame, "target", {"held"})
    assert peers.meter_id.tolist() == ["peer"]
    assert level == "L2_HARDWARE_REGION"
    output, reference = audit_targets(frame, ["target"], {"held"})
    assert output.peer_count.iloc[0] == 1
    assert set(reference.meter_id) == {"peer"}
    assert midrank_percentile(.5, [.1, .5, .9]) == .5


@pytest.mark.parametrize("algorithm", ["logistic", "hgb"])
def test_estimator_serialization_and_inference_batch_independence(tmp_path, algorithm):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"meter_id": [f"toy_{i}" for i in range(80)], "shape": rng.normal(size=80)})
    y = (frame["shape"] > 0).to_numpy(int)
    estimator = MixedEstimator(["shape"], [], algorithm, 17).fit(frame, y, np.ones(80))
    reference = estimator.predict_proba(frame)
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    joblib.dump({"selected_pu_rank": {"models": [estimator], "fit_meter_ids": frame.meter_id.tolist(), "rank_reference_scores": np.sort(reference)}}, bundle / "models/final_pu_channels.joblib")
    imputer = SimpleImputer().fit(frame[["shape"]])
    transformed = imputer.transform(frame[["shape"]])
    scaler = RobustScaler().fit(transformed)
    labels = ["complete", "equipment", "equipment_intermittent", "intermittent_use"]
    classifier = LogisticRegression(max_iter=200).fit(scaler.transform(transformed), np.tile(labels, 20))
    joblib.dump({"preprocessor": {"base_columns": ["shape"], "base_imputer": imputer, "base_scaler": scaler, "embedding_columns": []}, "labels": labels, "selected_method": "multinomial_l2_logistic", "logistic": classifier}, bundle / "models/conditional_subtype.joblib")
    all_path, one_path = tmp_path / "all.parquet", tmp_path / "one.parquet"
    frame.to_parquet(all_path, index=False)
    frame.iloc[[7]].to_parquet(one_path, index=False)
    full, single = score_feature_table(bundle, all_path), score_feature_table(bundle, one_path)
    np.testing.assert_allclose(full.iloc[7, 1:].to_numpy(float), single.iloc[0, 1:].to_numpy(float))
    restored = joblib.load(bundle / "models/final_pu_channels.joblib")["selected_pu_rank"]["models"][0]
    np.testing.assert_array_equal(reference, restored.predict_proba(frame))
