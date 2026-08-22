# Governance and Validation

## Data controls

- Only synthetic data is included in this public repository.
- Training data must contain a binary `alert` ground-truth field.
- Input schema and date parsing should be validated before feature generation.
- Real data should be access-controlled, encrypted, retained according to policy, and excluded from source control.

## Validation performed in this demo

- Stratified train/validation split to preserve class representation.
- Confusion matrix and class-level precision, recall, and F1.
- Learned operating threshold documented with model parameters.
- Rule-level contribution output for investigator review.
- Automated tests for feature creation, label requirements, training, scoring, and synthetic-data modes.

## Known limitations

- Synthetic labels are generated from a simple rule and do not represent investigator-adjudicated suspicious activity.
- Full-file aggregation can introduce temporal leakage.
- A single random split is insufficient for deployment approval.
- Logistic coefficients are model signals, not causal explanations.
- No production monitoring, calibration, fairness assessment, access control, case management, or independent validation is included.

## Production governance additions

A production model should have documented ownership, intended use, prohibited use, data lineage, versioned training sets, reproducible builds, approval gates, independent model-risk review, threshold rationale, drift monitoring, alert-volume monitoring, analyst feedback, rollback procedures, and periodic revalidation.
