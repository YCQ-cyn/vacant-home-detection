"""Explicit entry points for local recognition, training and audit workflows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="空置房识别 | Vacant-home recognition")
    result.add_argument("--workspace", type=Path, help="Private data/model workspace outside the source tree")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Display local workspace readiness without fitting")
    train = sub.add_parser("train", help="Run a named training stage on private inputs")
    train.add_argument("--stage", choices=["prepare", "ssl", "ablation", "validation", "report"], required=True)
    infer = sub.add_parser("infer", help="Frozen PU/subtype inference from a prepared feature table")
    infer.add_argument("--features", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    hp = sub.add_parser("high-power", help="Run the specialist branch explicitly")
    hp.add_argument("--fit", action="store_true", help="Fit and validate in a new private workspace")
    hp.add_argument("--audit-amplitude", action="store_true", help="Also run amplitude counterfactual inference")
    audit = sub.add_parser("peer-audit", help="Post-freeze matched-peer audit; no model fitting")
    audit.add_argument("--targets", nargs="+", required=True, help="Private meter identifiers supplied at runtime")
    audit.add_argument("--hardware", type=Path, required=True)
    audit.add_argument("--scores", type=Path, required=True)
    audit.add_argument("--holdout", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.workspace:
        os.environ["VACANT_HOME_WORKSPACE"] = str(args.workspace.resolve())
    # Imports are intentionally delayed until the workspace is configured.
    from .runtime import workspace
    root = workspace()
    if args.command == "status":
        names = ["fit_scope.csv", "future_human_holdout_manifest.csv", "models/final_pu_channels.joblib", "screening_results.csv"]
        print(json.dumps({"workspace": str(root), "ready": {name: (root / name).exists() for name in names}}, ensure_ascii=False, indent=2))
    elif args.command == "train":
        from .runtime import ensure_workspace
        ensure_workspace()
        if args.stage == "prepare":
            from .data import main as prepare
            if (root / "future_human_holdout_manifest.csv").exists():
                raise RuntimeError("Protected holdout already frozen; use a separate workspace to prepare another cohort")
            prepare()
        elif args.stage == "ssl":
            from .representation import run_research
            run_research()
        elif args.stage == "ablation":
            from .models import run_ablation
            run_ablation()
        elif args.stage == "validation":
            from .validation import run_validation
            run_validation()
        elif args.stage == "report":
            from .reporting import write_local_report
            write_local_report()
    elif args.command == "infer":
        from .inference import score_feature_table
        result = score_feature_table(root, args.features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Inference complete: {len(result)} users")
    elif args.command == "high-power":
        if not args.fit:
            raise RuntimeError("Existing branch is frozen. Pass --fit only for an explicitly separate training workspace")
        from .high_power.runner import run_branch
        run_branch(audit_amplitude=args.audit_amplitude)
    elif args.command == "peer-audit":
        from .peer_audit import run_audit
        run_audit(args.scores, args.hardware, args.holdout, args.targets, args.output_dir)

