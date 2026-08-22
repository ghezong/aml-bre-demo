import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aml_core.aml_rules import build_rule_features


RULE_FEATURE_COLUMNS = [
    "rule_many_to_one_count",
    "rule_one_to_many_count",
    "rule_mule_pair_total",
    "rule_dormant_gap_days",
]


def preprocess_rule_features(transactions, require_label=False):
    features_df = build_rule_features(transactions)
    X = features_df[RULE_FEATURE_COLUMNS]
    y = features_df["alert"] if "alert" in features_df.columns else None

    if require_label and y is None:
        raise ValueError("Training dataset must include an 'alert' column.")
    return X, y, features_df


def _best_f1_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    for idx, threshold in enumerate(thresholds):
        p = precision[idx]
        r = recall[idx]
        if (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def train_rule_model(transactions):
    X, y, feature_df = preprocess_rule_features(transactions, require_label=True)

    if y.nunique() < 2:
        raise ValueError("Training labels in 'alert' must contain both 0 and 1 classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    decision_threshold = _best_f1_threshold(y_test, y_proba)
    y_pred = (y_proba >= decision_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4)

    clf = pipeline.named_steps["clf"]
    coefficients = dict(zip(RULE_FEATURE_COLUMNS, clf.coef_[0].tolist()))
    positive_slice = feature_df[feature_df["alert"] == 1]
    rule_reference_levels = {
        col: float(positive_slice[col].median()) if not positive_slice.empty else 0.0
        for col in RULE_FEATURE_COLUMNS
    }

    learned_params = {
        "decision_threshold": float(decision_threshold),
        "coefficients": coefficients,
        "rule_reference_levels_median_alerts": rule_reference_levels,
    }

    eval_results = {
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }

    return {
        "pipeline": pipeline,
        "rule_features": RULE_FEATURE_COLUMNS,
        "learned_params": learned_params,
        "evaluation": eval_results,
    }


def predict_with_rule_model(model_bundle, transactions):
    X, _, feature_df = preprocess_rule_features(transactions, require_label=False)
    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["learned_params"]["decision_threshold"]

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = feature_df.copy()
    result["prediction_probability"] = probabilities
    result["prediction"] = predictions

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    x_scaled = scaler.transform(X)
    coefficients = clf.coef_[0]

    contribution_columns = []
    for idx, col in enumerate(RULE_FEATURE_COLUMNS):
        contrib_col = f"contrib_{col}"
        result[contrib_col] = x_scaled[:, idx] * coefficients[idx]
        contribution_columns.append(contrib_col)

    result["top_rule_contributor"] = (
        result[contribution_columns].abs().idxmax(axis=1).str.replace("contrib_rule_", "", regex=False)
    )
    result["top_rule_contribution_score"] = result[contribution_columns].abs().max(axis=1)
    return result


def save_model(model_bundle, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, filename)


def load_model(filename):
    return joblib.load(filename)


def default_paths(project_root):
    root = Path(project_root)
    return {
        "training_file": root / "aml_app" / "data" / "training" / "training_rules.csv",
        "new_file": root / "aml_app" / "data" / "new" / "new_rules.csv",
        "model_file": root / "aml_app" / "models" / "aml_model.pkl",
        "scored_output": root / "aml_app" / "outputs" / "scored_predictions.csv",
    }


def main():
    parser = argparse.ArgumentParser(description="Train and apply AML rule-based ML model.")
    parser.add_argument("--training_file", type=str, default=None)
    parser.add_argument("--new_file", type=str, default=None)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--scored_output", type=str, default=None)
    parser.add_argument("--project_root", type=str, default=None)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = Path(args.project_root) if args.project_root else script_path.parents[2]
    defaults = default_paths(project_root)

    training_file = Path(args.training_file) if args.training_file else defaults["training_file"]
    new_file = Path(args.new_file) if args.new_file else defaults["new_file"]
    model_file = Path(args.model_file) if args.model_file else defaults["model_file"]
    scored_output = Path(args.scored_output) if args.scored_output else defaults["scored_output"]

    transactions = pd.read_csv(training_file)
    model_bundle = train_rule_model(transactions)
    save_model(model_bundle, model_file)

    print("Confusion Matrix:")
    print(model_bundle["evaluation"]["confusion_matrix"])
    print("Classification Report:")
    print(model_bundle["evaluation"]["classification_report"])
    print("Learned rule parameters:")
    print(model_bundle["learned_params"])

    if new_file.exists():
        new_data = pd.read_csv(new_file)
        scored = predict_with_rule_model(model_bundle, new_data)
        scored_output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(scored_output, index=False)
        print(f"Scored output saved to {scored_output}")


if __name__ == "__main__":
    main()
