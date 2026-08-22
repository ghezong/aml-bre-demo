# Methodology

## Feature Engineering

Each transaction is represented using four interpretable signals:

1. `rule_many_to_one_count`: unique senders associated with the receiving account.
2. `rule_one_to_many_count`: unique receivers associated with the sending account.
3. `rule_mule_pair_total`: total amount exchanged by the sender/receiver pair.
4. `rule_dormant_gap_days`: elapsed days since the sender's previous transaction.

The prototype computes these features over the supplied file. A production implementation must use time-bounded, as-of windows so future transactions cannot influence an earlier decision.

## Model

The model is a class-balanced logistic regression preceded by `StandardScaler`. Logistic regression was selected as a transparent baseline for tabular data: coefficients show directional association and standardized features allow approximate comparison of rule influence. It is not intended to establish causation.

The data is split into stratified training and validation partitions. The operating probability threshold is selected on validation predictions by maximizing F1. This makes the threshold a learned artifact rather than a manually entered value, while still requiring a business decision about the desired precision/recall tradeoff.

## Evaluation

The workflow reports:

- confusion matrix;
- precision, recall, and F1 by class;
- learned probability threshold;
- rule coefficients and alert-group reference levels;
- per-row rule contributions for scored data.

Because the included labels are synthetic, metrics are a demonstration of the evaluation pipeline and must not be interpreted as evidence of production accuracy.
