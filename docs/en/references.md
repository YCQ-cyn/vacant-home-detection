# Method References

[中文](../zh/references.md) | **English**

These references explain the methods and their business motivation. The literature is not evidence of this project's vacancy recognition accuracy.

## Positive-unlabeled recognition

### Bagging PU

- Mordelet, F. and Vert, J.-P. *A bagging SVM to learn from positive and unlabeled examples*. Pattern Recognition Letters 37 (2014), 201-209. DOI: https://doi.org/10.1016/j.patrec.2013.06.007
- Problem addressed: U contains both active and unrecorded-vacancy users, so a single P-versus-all-U classifier is sensitive to which U cases are treated as comparison examples.
- Application: bagged U subsampling provides the PU comparison mechanism, using HGB and logistic heads. Sampled U receives an internal comparison target only inside each bag and remains unlabeled.

### SCAR, SAR and propensity-aware PU

- Bekker, J. and Davis, J. *Learning from Positive and Unlabeled Data under the Selected At Random Assumption*. arXiv:1808.08755 and later journal/conference versions: https://arxiv.org/abs/1808.08755
- Problem addressed: Legacy positives were selected because some curve patterns were easier for experts to recognize. Selection probability therefore plausibly depends on zero fraction, long silence, magnitude and obvious intermittency.
- Application: a historical-positive selection propensity model reweights labeled positives. Its inputs are audited separately from the vacancy score. P+U alone cannot fully identify the labeling mechanism.

### nnPU

- Kiryo, R. et al. *Positive-Unlabeled Learning with Non-Negative Risk Estimator*. NeurIPS 2017. Official proceedings: https://papers.nips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html ; arXiv: https://arxiv.org/abs/1703.00593
- Problem addressed: unbiased PU risk can become negative and overfit with flexible models.
- Application: non-negative PU risk controls negative-risk overfitting. Uncertain class priors require sensitivity checks, and outputs are not calibrated vacancy probabilities.

### Class prior

- Elkan, C. and Noto, K. *Learning classifiers from only positive and unlabeled data*. KDD 2008. DOI: https://doi.org/10.1145/1401890.1401920
- du Plessis, M., Niu, G. and Sugiyama, M. *Class-prior estimation for learning from positive and unlabeled data*. Machine Learning 2017. DOI: https://doi.org/10.1007/s10994-016-5589-8
- Application: prior values are sensitivity/regularization settings. They do not estimate population vacancy prevalence under feature-dependent historical labeling.

## Time-series representation

### Dilated TCN

- Bai, S., Kolter, J. Z. and Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271: https://arxiv.org/abs/1803.01271
- Problem addressed: learn multi-scale daily and 42-day structure with a compact receptive field and stable training.
- Method: compact residual dilated Conv1D blocks with dilations 1, 2, 4, 8 encode multi-scale behavior.

### TS2Vec-style hierarchical contextual contrast

- Yue, Z. et al. *TS2Vec: Towards Universal Representation of Time Series*. AAAI 2022. DOI: https://doi.org/10.1609/aaai.v36i8.20881 ; official authors' code: https://github.com/zhihanyue/ts2vec
- Problem addressed: masked reconstruction alone can learn local amplitude shortcuts and may fail to organize behaviorally similar contexts.
- Method: hierarchical contextual contrast uses aligned masked views to preserve clock-time information.

## Smart-meter and occupancy context

- Kleiminger, W. et al. *Occupancy Detection from Electricity Consumption Data*. BuildSys 2013. DOI: https://doi.org/10.1145/2528282.2528295
- UK-DALE: Kelly, J. and Knottenbelt, W. *The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes*. Scientific Data 2, 150007 (2015). DOI: https://doi.org/10.1038/sdata.2015.7
- REFIT: Murray, D., Stankovic, L. and Stankovic, V. *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. Scientific Data 4, 160122 (2017). DOI: https://doi.org/10.1038/sdata.2016.122 ; dataset: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
- REDD: Kolter, J. Z. and Johnson, M. J. *REDD: A public data set for energy disaggregation research*. SustKDD 2011. Project reference: http://redd.csail.mit.edu/
- Pecan Street Dataport: https://www.pecanstreet.org/dataport/
- Application: these sources support shape, transitions, recurrence and long-horizon behavior modeling. Their data are not included in training because sampling semantics, geography, household behavior, meter definitions and access terms differ.

## Active learning and protected review

- Settles, B. *Active Learning Literature Survey*. University of Wisconsin-Madison, 2009: https://minds.wisconsin.edu/handle/1793/60660
- Problem addressed: choosing only top scores under a limited review budget biases validation.
- Application: combine stratified random blind review with diverse targeted challenges. Random review supports population assessment; targeted review identifies failure cases and does not provide population-unbiased accuracy.

## Point-semantics boundary

Periodic frozen-power sampling can miss and alias short events. Full-interval averaging and amplitude dilution cannot be assumed without device-specific evidence. Runs are sample counts; only cumulative-register differences establish true energy.
