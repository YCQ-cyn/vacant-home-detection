# Reproducibility

[中文](../zh/reproducibility.md) | **English**

## Source-only verification

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m vacant_home --help
```

Tests use programmatically generated arrays and toy identifiers. They check routing gates, missing-family abstention, scale counterfactual semantics, protected-user sampling, matched-peer isolation, local rank references, estimator serialization and encoder shape contracts. They do not measure real vacancy accuracy.

## Private cohort reproduction

Supply the prepared private inputs listed in [data_contract.md](data_contract.md). Run in a **new** workspace. Existing frozen models and review-pool manifests are kept intact.

```powershell
$env:VACANT_HOME_WORKSPACE = 'D:\Private\vacant-home-training'
python -m vacant_home train --stage prepare
python -m vacant_home train --stage ssl
python -m vacant_home train --stage ablation
python -m vacant_home train --stage validation
python -m vacant_home train --stage report
python -m vacant_home high-power --fit --audit-amplitude
```

CPU training can take substantial time. These commands are explicit training operations; inference and peer audit never start them automatically.

Fixed seeds, subtype balancing, medium-support weights, prior settings and engineering Tier gates are recorded in source and the specialist decision specification. Daily rows remain grouped by meter for supervision. Protected users never enter self-supervised fitting or cohort/prototype references.

PU evaluation is label-cross-fitted. Research representations are transductive within the permitted research pool, so daily-embedding representation gains are exploratory; they must not be upgraded to fully fold-isolated inductive claims.

## Frozen inference

```powershell
python -m vacant_home --workspace 'D:\Private\vacant-home-trained' infer --features 'D:\Private\prepared_features.parquet' --output 'D:\Private\pu_subtype_scores.csv'
```

Rank references come from the fitted bundle and stay fixed under inference batch changes. Older locally supplied bundles without saved references require the full fitted cohort in the feature table; the function raises an error instead of silently ranking against only a few new users.

## Post-freeze peer audit

Supply target identifiers at runtime rather than embedding real users in source. `hardware.csv` columns: meter ID, utility, meter vendor, meter version and terminal vendor. `scores.csv` contains the specialist metrics plus quality and power-index fields. `holdout.csv` supplies the entire protected ID set.

```powershell
python -m vacant_home peer-audit --targets YOUR_PRIVATE_TARGET --hardware 'D:\Private\hardware.csv' --scores 'D:\Private\scores.csv' --holdout 'D:\Private\holdout.csv' --output-dir 'D:\Private\peer-audit'
```

The audit freezes and hashes its matching specification before reading target metrics. It preserves model scores/Tiers and only computes matched-peer percentiles.
