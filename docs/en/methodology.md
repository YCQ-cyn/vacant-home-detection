# Methodology

[中文](../zh/methodology.md) | **English**

## Recognition

1. Route invalid data and valid electrical silence before behavioral modeling.
2. Encode daily periodic power samples with compact residual dilated convolutions.
3. Fuse daily embeddings with sampled-power quantiles, shape similarity, recurrence, transitions, sample runs, weekly structure, episode sparsity and state changes.
4. Fit bagged P-versus-sampled-U comparison heads with low comparison weights. Keep U explicitly unlabeled and calibrate only evidence ranks against a frozen allowed reference cohort.
5. Use independent automatic/human evidence axes, conditional subtype models and explicit conflict/abstention routes.

## High-power discovery

Power qualification uses the median of four allowed-reference empirical percentiles: raw median, q90, baseline floor and the fraction of frozen samples above 1kW. Relative high power starts at 0.75 and strict high power at 0.90.

Persistent load uses sample-based routes: median/fraction/high-load-day persistence, or baseline/fraction/daily-q90 persistence. These are qualification tests, not vacancy evidence. The funnel's persistent count is the intersection with relative high power; it can differ from the total persistent-load population.

Three evidence families are aggregated by robust medians:

- Human absence: Human and Active evidence opposition conditioned on comparable magnitude/baseline.
- Automatic structure: automatic compatibility, recurring background, temporal structure and transition support.
- Learned representation: normalized Behavior embeddings, separate equipment-only/equipment-intermittent similarity and a behavior-only PU comparison channel.

Tier A requires positive valid support from all three families plus pre-registered stability and quality gates. Missing families cannot be replaced by extra correlated votes. The thresholds are engineering gates without Gold calibration; they are not optimal accuracy thresholds.

## Validation

Use positive label cross-fitting, positive-versus-power-matched-U rank advantage with bootstrap intervals, cross-fitted fake-positive permutation controls, leave-subtype-out, absolute high-power/low-power-rich-activity challenge cohorts, window/seed/prior sensitivity and amplitude counterfactuals.

No fresh negative Gold labels exist. Historical recovery and P-versus-U separation can reward selection bias; neither supplies true classification accuracy. The implementation and reports disclose research-pool transductive representation fitting.

## Matched peers

Before inspecting target percentiles, freeze matching metadata and reference rules. Match utility, meter vendor and meter version; keep terminal vendor exact when at least 10 peers exist. Otherwise disclose the terminal fallback. Choose at most 30 nearest frozen power-index peers; never use candidate scores to pick neighbors. Exclude the target and every future holdout from reference distributions.
