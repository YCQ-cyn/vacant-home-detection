# Recognition Research Notes

[中文](../zh/research_notes.md) | **English**

Access date: 2026-09-26. These notes record methods considered for the first complete candidate. They are not evidence of physical vacancy accuracy.

## Positive-unlabeled recognition

### Bagging PU

- Mordelet, F. and Vert, J.-P. *A bagging SVM to learn from positive and unlabeled examples*. Pattern Recognition Letters 37 (2014), 201-209. DOI: https://doi.org/10.1016/j.patrec.2013.06.007
- Problem addressed: U contains both active and unrecorded-vacancy users, so a single P-versus-all-U classifier is sensitive to which U cases are treated as comparison examples.
- Decision: adopt bagged U subsampling as the primary auditable PU mechanism, using HGB and logistic heads. Sampled U receives an internal comparison target only inside each bag; it is never described as a confirmed negative.

### SCAR, SAR and propensity-aware PU

- Bekker, J. and Davis, J. *Learning from Positive and Unlabeled Data under the Selected At Random Assumption*. arXiv:1808.08755 and later journal/conference versions: https://arxiv.org/abs/1808.08755
- Problem addressed: Legacy positives were selected because some curve patterns were easier for experts to recognize. Selection probability therefore plausibly depends on zero fraction, long silence, magnitude and obvious intermittency.
- Decision: SCAR is not assumed. A separate, explicitly named Legacy-selection propensity model is used only to reweight labeled positives. Propensity inputs are audited and never passed to the vacancy head as a score. Identifiability is still incomplete with P+U alone, so SAR is an ablation rather than unquestioned truth.

### nnPU

- Kiryo, R. et al. *Positive-Unlabeled Learning with Non-Negative Risk Estimator*. NeurIPS 2017. Official proceedings: https://papers.nips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html ; arXiv: https://arxiv.org/abs/1703.00593
- Problem addressed: unbiased PU risk can become negative and overfit with flexible models.
- Decision: implement a small 64-32-1 MLP as sensitivity only. Its class prior is uncertain under SAR, so it cannot be the sole primary model and its output is not a probability.

### Class prior

- Elkan, C. and Noto, K. *Learning classifiers from only positive and unlabeled data*. KDD 2008. DOI: https://doi.org/10.1145/1401890.1401920
- du Plessis, M., Niu, G. and Sugiyama, M. *Class-prior estimation for learning from positive and unlabeled data*. Machine Learning 2017. DOI: https://doi.org/10.1007/s10994-016-5589-8
- Decision: prior values are sensitivity/regularization settings only. They are not interpreted as the population vacancy prevalence because Legacy labeling is feature-dependent.
- Implementation audit: an earlier draft floored the U comparison weight at 0.12, making prior settings 0.06 and 0.10 identical. Recognition uses a continuous weight scaling centered at 0.10 so every prior setting is a real perturbation. The resulting sensitivity is still not a class-prior estimate.

## Time-series representation

### Dilated TCN

- Bai, S., Kolter, J. Z. and Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271: https://arxiv.org/abs/1803.01271
- Problem addressed: learn multi-scale daily and 42-day structure with a compact receptive field and stable training.
- Decision: retain residual dilated Conv1D blocks with dilations 1,2,4,8. Compare hidden widths 32 and 64 only; no broad architecture search.

### TS2Vec-style hierarchical contextual contrast

- Yue, Z. et al. *TS2Vec: Towards Universal Representation of Time Series*. AAAI 2022. DOI: https://doi.org/10.1609/aaai.v36i8.20881 ; official authors' code: https://github.com/zhihanyue/ts2vec
- Problem addressed: masked reconstruction alone can learn local amplitude shortcuts and may fail to organize behaviorally similar contexts.
- Decision: test a bounded TS2Vec-style objective with instance and timestamp contrast at multiple max-pooled scales. It uses aligned masked views only. Time shifts, warps and shuffles are rejected because clock time is business information. This implementation is described as “TS2Vec-style”, not as a byte-for-byte reproduction.

### Long-horizon encoder

- Decision: compare a small Temporal TCN against a one-layer GRU. Tasks are masked-day prediction, 14/21/28/42-day consistency and temporal-order discrimination. A Transformer is rejected for the first model because there are only 1,827 user sequences and no demonstrated failure requiring its larger search space.

## Smart-meter and occupancy context

- Kleiminger, W. et al. *Occupancy Detection from Electricity Consumption Data*. BuildSys 2013. DOI: https://doi.org/10.1145/2528282.2528295
- UK-DALE: Kelly, J. and Knottenbelt, W. *The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes*. Scientific Data 2, 150007 (2015). DOI: https://doi.org/10.1038/sdata.2015.7
- REFIT: Murray, D., Stankovic, L. and Stankovic, V. *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. Scientific Data 4, 160122 (2017). DOI: https://doi.org/10.1038/sdata.2016.122 ; dataset: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
- REDD: Kolter, J. Z. and Johnson, M. J. *REDD: A public data set for energy disaggregation research*. SustKDD 2011. Project reference: http://redd.csail.mit.edu/
- Pecan Street Dataport: https://www.pecanstreet.org/dataport/
- Decision: these sources support the use of shape, transitions, recurrence and long-horizon behavior, but they are not mixed into training. Sampling semantics, geography, household behavior, submeter/whole-home definitions and access terms differ from the current 15-minute periodic frozen-power samples.

## Active learning and protected review

- Settles, B. *Active Learning Literature Survey*. University of Wisconsin-Madison, 2009: https://minds.wisconsin.edu/handle/1793/60660
- Problem addressed: if only 10-20 future labels are approved, choosing only top scores wastes review budget and biases validation.
- Decision: freeze two groups before Recognition fitting: 10 fixed-seed stratified RANDOM_BLIND cases and 20 diversity-aware TARGETED_CHALLENGE cases. The targeted set combines disagreement, conflict, high-power automatic-compatible, low-power rich-activity, smooth modulation and manifold novelty. It is for failure discovery, never population-unbiased accuracy.

## Business-physics source

The private historical Word report supplies observations about resident baseload, automatic equipment backgrounds, sparse return episodes, low-load inhabited homes and weekend use. Its fixed daily-energy and 0.5kW rules are not replicated as final decisions. The report itself is not distributed.

## Point-semantics boundary

Periodic frozen-power sampling can miss and alias short events. Full-interval averaging and amplitude dilution cannot be assumed without device-specific evidence. Runs are sample counts; only cumulative-register differences establish true energy.

## Representation implementation audit

- During the first full training attempt, the amplitude-scaling augmentation changed only `signed_log_power`, while the behavior branch consumed `signed_log_difference` rather than that absolute-power channel. The nominal behavior-invariance loss was therefore largely vacuous and partly confounded with masking.
- The incomplete attempt was stopped. The corrected implementation recomputes `signed_log_difference` after scaling and compares unmasked original versus scaled behavior embeddings. All research representations used for model selection are retrained from the frozen scope after this repair.
