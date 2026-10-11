from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax


def empirical_rank(values, reference):
    reference = np.sort(np.asarray(reference, float))
    return np.searchsorted(reference, np.asarray(values, float), side="right") / max(1, len(reference))


def score_feature_table(bundle_root, feature_path):
    bundle_root = Path(bundle_root)
    frame = pd.read_parquet(feature_path)
    artifact = joblib.load(bundle_root / "models" / "final_pu_channels.joblib")
    selected = artifact["selected_pu_rank"]
    models = selected["models"]
    score = np.mean([model.predict_proba(frame) for model in models], axis=0)
    reference = selected.get("rank_reference_scores")
    if reference is None:
        fit_ids = set(selected["fit_meter_ids"])
        if not fit_ids.issubset(set(frame.meter_id.astype(str))):
            raise ValueError("Bundle lacks frozen rank references; supply the complete fitted-cohort feature table")
        fit_mask = frame.meter_id.astype(str).isin(fit_ids)
        reference = score[fit_mask]
    rank = empirical_rank(score, reference)
    subtype = joblib.load(bundle_root / "models" / "conditional_subtype.joblib")
    preprocessor = subtype["preprocessor"]
    transformed = preprocessor["base_scaler"].transform(
        preprocessor["base_imputer"].transform(frame[preprocessor["base_columns"]])
    )
    if preprocessor["embedding_columns"]:
        embedding = preprocessor["embedding_scaler"].transform(
            preprocessor["embedding_imputer"].transform(frame[preprocessor["embedding_columns"]])
        )
        transformed = np.c_[transformed, preprocessor["pca"].transform(embedding)]
    labels = subtype["labels"]
    if subtype["selected_method"] == "multinomial_l2_logistic":
        classifier = subtype["logistic"]
        raw = classifier.predict_proba(transformed)
        support = raw[:, [list(classifier.classes_).index(label) for label in labels]]
    else:
        raw = np.zeros((len(frame), len(labels)))
        for index, label in enumerate(labels):
            prototype = subtype["prototypes"][label]
            centers = np.asarray(prototype["centers"])
            distance = np.linalg.norm(transformed[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
            raw[:, index] = np.exp(-distance / float(prototype["scale"]))
        support = softmax(raw, axis=1)
    output = pd.DataFrame({
        "meter_id": frame.meter_id.astype(str),
        "selected_pu_evidence_score": score,
        "selected_pu_evidence_rank": rank,
    })
    for index, label in enumerate(labels):
        output[f"conditional_subtype_support_{label}"] = support[:, index]
    return output


def main():
    parser = argparse.ArgumentParser()
    from .runtime import workspace
    parser.add_argument("--bundle-root", default=str(workspace()))
    parser.add_argument("--features", default="current_cohort_feature_table.parquet")
    parser.add_argument("--output", default="rescored_current_cohort.csv")
    args = parser.parse_args()
    root = Path(args.bundle_root)
    features = Path(args.features)
    if not features.is_absolute():
        features = root / features
    output = score_feature_table(root, features)
    target = Path(args.output)
    if not target.is_absolute():
        target = root / target
    output.to_csv(target, index=False)
    print(target)


if __name__ == "__main__":
    main()
