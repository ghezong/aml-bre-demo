import pandas as pd

from src.aml_core.aml_rules import build_rule_features
from src.aml_core.generate_sample_data import generate_transaction_data
from src.aml_core.ml_model import predict_with_rule_model, train_rule_model


def test_training_generator_includes_binary_alert():
    data = generate_transaction_data(100, "training")
    assert "alert" in data.columns
    assert set(data["alert"].unique()).issubset({0, 1})


def test_new_generator_excludes_alert():
    data = generate_transaction_data(100, "new")
    assert "alert" not in data.columns


def test_rule_features_have_expected_columns():
    data = generate_transaction_data(100, "training")
    features = build_rule_features(data)
    expected = {
        "rule_many_to_one_count",
        "rule_one_to_many_count",
        "rule_mule_pair_total",
        "rule_dormant_gap_days",
    }
    assert expected.issubset(features.columns)


def test_model_trains_and_scores_new_data():
    training = generate_transaction_data(300, "training")
    new_data = generate_transaction_data(40, "new")
    bundle = train_rule_model(training)
    scored = predict_with_rule_model(bundle, new_data)
    assert len(scored) == len(new_data)
    assert {"prediction", "prediction_probability", "top_rule_contributor"}.issubset(scored.columns)
    assert set(scored["prediction"].unique()).issubset({0, 1})


def test_training_requires_alert_label():
    new_data = generate_transaction_data(100, "new")
    try:
        train_rule_model(new_data)
    except ValueError as error:
        assert "alert" in str(error)
    else:
        raise AssertionError("Expected unlabeled data to be rejected")
