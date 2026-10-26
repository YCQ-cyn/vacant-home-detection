# High-Power Branch Decision Specification

- Rule policy: fixed before candidate scoring
- Stage: `HIGH_POWER_VACANCY_DISCOVERY`
- Point semantics: 15-minute periodic frozen active-power samples（15分钟周期冻结有功功率样本）
- Gold or new human labels used: no

## Immutable upstream and identity rules

- The frozen recognition workspace is read-only to the specialist branch. Its models, scores and review manifests must remain intact.
- Legacy source has 137 rows and 136 unique meter IDs. The duplicated meter occurs twice and receives one unit of weight.
- The unique subtype counts must be: intermittent-use 62, equipment 29, complete 24, equipment-intermittent 21. A mismatch is a hard failure.
- All prototype, similarity, and PU fitting operates on unique `meter_id` values.
- The 30 frozen `FUTURE_HUMAN_HOLDOUT` meters are inference-only. They do not participate in fit, scaler/PCA, percentiles, prototypes, thresholds, rule design, candidate-score design, or model selection.

## Reference populations

- `fit_reference`: valid, non-strict-zero, non-holdout users represented in the frozen Recognition fit scope.
- `high_power_reference`: members of `fit_reference` whose pre-registered relative high-power flag is true and whose persistent-high-load flag is true.
- Percentiles for holdout users are empirical-CDF inference against the already frozen non-holdout reference. Holdout users never alter a percentile denominator or cutoff.
- Equipment references are 29 unique `equipment` Legacy users. Equipment-intermittent references are 21 unique `equipment_intermittent` Legacy users. They are never merged into one prototype.

## High-power qualification; not vacancy evidence

For each user, four magnitude views are calculated from finite samples:

1. raw sample median;
2. raw sample q90;
3. baseline floor (median of daily sample q10 values);
4. fraction of frozen samples exceeding 1 kW.

`power_index` is the median of the four empirical percentile ranks against `fit_reference`.

- Relative high power: `power_index >= 0.75`.
- Strict high power: `power_index >= 0.90`.
- Persistent high load is true when either:
  - sample median >= 1 kW, fraction of frozen samples >1 kW >= 0.50, and at least 21 daily sample medians exceed 1 kW; or
  - baseline-floor percentile >= 0.90, >1 kW frozen-sample-fraction percentile >= 0.75, and at least 21 daily sample q90 values exceed 2 kW.
- An occasional-peak case has high q90 or maximum but fails persistent-high-load. It cannot enter the persistent high-power vacancy candidate set.

The fractions above describe observed frozen samples, not physical runtime. High power is only cohort eligibility and is neither vacancy nor active/occupied evidence.

## Three evidence families

The branch reports several channels, but they are organized into three correlated evidence families and are not advertised as independent evidence.

### Family 1: absence of sustained human interaction

- Power-conditioned Human-Interaction Evidence opposition.
- Power-conditioned Active Evidence opposition.
- Conditioning uses nearest non-holdout reference users in log-median, log-q90, log-baseline and >1 kW frozen-sample-fraction space.
- Family support is the conservative median of the two opposition percentiles.

### Family 2: automatic/background structure

- Automatic-Control Compatibility.
- Cross-day normalized shape stability, daily trajectory similarity, background recurrence, temporal stability, lag-7 structure and sample-state/episode persistence.
- Equipment conditional subtype support can provide positive context or abstain. Legacy raw magnitude distance cannot veto a user.
- This family means compatibility with sustained automatic/background mechanisms; it does not identify a device or prove absence of people.

### Family 3: magnitude-resistant learned representation

- The first 32 dimensions of each frozen Daily Dual-TCN are the Behavior branch. Each seed is unit-normalized before similarity or PU fitting.
- `equipment_only_similarity` is calculated only against the 29 unique equipment Legacy users.
- `equipment_intermit_similarity` is calculated only against the 21 unique equipment-intermittent Legacy users.
- A behavior-only Bagging PU channel excludes raw magnitude fields and never interprets U as known normal.
- The two similarities remain separate in every output.

## Score aggregation and missing channels

- Every channel is converted to an empirical percentile against the frozen non-holdout `high_power_reference` when that reference is applicable.
- A family is valid only when at least two of its declared channels are finite. Otherwise it abstains.
- The high-power branch score is the median of the three valid family supports.
- Tier A requires all three families to be valid and positively supportive. Correlated subchannels cannot substitute for a missing family.
- Missing data never receives a neutral positive value. It produces abstention and can only lower the achievable tier.

## Pre-registered engineering thresholds

These are no-Gold engineering gates. They are not optimized thresholds, calibrated probabilities, or accuracy thresholds.

### Tier A: core high-power suspected-vacancy specialist candidate

All conditions are required:

- valid data, relative high power, and persistent high load;
- all three evidence families valid;
- each family support >= 0.70 (top 30% support within the high-power reference);
- human/active opposition family >= 0.65 (top 35% opposition);
- `equipment_only_similarity` percentile >= 0.70;
- automatic-structure support or equipment subtype support >= 0.60 (top 40%);
- perturbation/stability median conditional rank >= 0.85 and rank p10 >= 0.70;
- no severe missingness/acquisition anomaly and no isolated-peak-only pattern.

### Tier B: secondary high-power review candidate

- valid data, relative high power, and persistent high load;
- at least two evidence families valid and >= 0.65;
- human/active opposition >= 0.50;
- stability median conditional rank >= 0.70;
- no severe quality anomaly or isolated-peak-only pattern.

Tier B remains `暂无法可靠判断` in the combined business result unless independently routed by the frozen Recognition model.

### Tier C: audited high-power cohort

All remaining relative/strict/persistent high-power users. Tier C is not a vacancy candidate label.

Only Tier A receives the added business class `高功率疑似空置专项候选`. This is a screening candidate, not confirmed vacancy.

## Stability grid

Candidate evidence is recomputed or audited across:

- seeds: 20260926, 20260927, 20260928;
- windows: 14, 21, 28, 42 days;
- PU prior sensitivities: 0.06, 0.10, 0.16 (regularization only; not population prevalence);
- medium-positive weights: 0.25, 0.50, 0.75;
- scoring variants: full, no-PU, no-subtype, no-TCN;
- reference variants: equipment-only and equipment plus equipment-intermittent, while retaining separate similarity fields.

## Amplitude-scaling audit

For every one of the 29 unique equipment Legacy users, preserve the exact temporal structure and multiply all finite power samples by 1, 2, 5 and 10. Report:

- frozen Recognition global score and rank;
- Daily-TCN Behavior embedding cosine to the x1 embedding;
- new high-power branch score;
- new high-power conditional rank.

The pre-registered audit checks are Behavior cosine median >=0.95 and p10 >=0.85, plus x10 branch-score median decline <=0.10. These checks are diagnostics only and will not be changed after results are seen.

## Quality and confounding review

Every Tier A/B row reports missingness, negative/unresolved samples, peak concentration, high-load-day persistence, utility, meter/vendor metadata when available, and flags for possible non-residential/shared/special-service use. Metadata-based special-use flags are confounders, not automatic exclusions unless data integrity is invalid.

## Output interpretation

- The branch finds and ranks all qualifying users in the full input cohort.
- Future holdout users may appear only after the complete rule and model artifacts are frozen. Their inference is clearly marked.
- No output is a vacancy probability, confirmed vacancy, Gold accuracy, precision, recall, or F1.
