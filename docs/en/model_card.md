# Model card

[中文](../zh/model_card.md) | **English**

## Intended use

Long-horizon residential vacancy screening, ranking, subtype evidence and review prioritization. Site verification remains necessary for business truth.

## Selected implementation

Dual-branch daily TCN + semantics-robust features + Bagging PU HGB. Behavior and magnitude branches each emit 32 dimensions. Dilations are 1, 2, 4, 8.

## Supervision

Historical expert-confirmed positives are deduplicated per meter. Four subtype counts are 62 intermittent-use, 29 equipment-maintained, 24 complete and 21 equipment-intermittent. Current-window HIGH/MEDIUM/LOW support is handled separately. HIGH is hard positive, MEDIUM uses soft weighting, LOW is diagnostic in the main head. Some specialist sensitivity code uses a documented low evidence weight; it does not promote LOW to hard labels.

## Sample semantics

15-minute periodic frozen active-power samples. Snapshot/short-window-average/full-interval-average interpretations are unresolved without device mapping evidence. Features avoid true-kWh assumptions and express runs as consecutive sample counts.

## Protection

Thirty future-review meters were frozen before fitting: 10 stratified RANDOM_BLIND and 20 TARGETED_CHALLENGE. Their curves do not enter SSL, scalers, PCA, prototypes, PU/propensity fitting, model selection, thresholds, or high-power references. Targeted selection does not yield a population-unbiased accuracy sample.

## Evaluation qualification

PU labels are held out by meter. Research SSL representations and rank transforms were fitted jointly on the allowed research pool. This is label-cross-fitted, research-pool transductive evidence; it is not fully fold-isolated user-level representation validation. The direct-feature comparator is the cleaner strict reference.

Historical positive selection heavily underrepresents sustained high-load vacancy. The specialist branch addresses that failure mode, but candidates remain unconfirmed. Automatic compatibility is curve compatibility, not proof of an appliance or environmental driver.

## Output interpretation

Routes: high vacancy suspicion, moderate vacancy suspicion, likely active use, uncertain and invalid data. Specialist Tier A is a high-power vacancy screening candidate, with subtype reported separately.

Behavior-only PU raw scores are not guaranteed to be amplitude-invariant.

## Distribution

This repository contains methods and tests only. Private curves, labels, meter identifiers, case plots, fitted reference distributions and trained weights must be supplied locally. No real Accuracy, Precision, Recall, F1, population vacancy rate or calibrated vacancy probability is asserted.
