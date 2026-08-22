import pandas as pd


def _prepare_transactions(transactions):
    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    return tx


def build_rule_features(transactions):
    tx = _prepare_transactions(transactions)

    many_to_one = (
        tx.groupby("receiver_account")["sender_account"]
        .nunique()
        .rename("rule_many_to_one_count")
        .reset_index()
    )

    one_to_many = (
        tx.groupby("sender_account")["receiver_account"]
        .nunique()
        .rename("rule_one_to_many_count")
        .reset_index()
    )

    mule_pair_total = (
        tx.groupby(["sender_account", "receiver_account"])["amount"]
        .sum()
        .rename("rule_mule_pair_total")
        .reset_index()
    )

    tx = tx.sort_values(["sender_account", "transaction_date"])
    tx["rule_dormant_gap_days"] = (
        tx.groupby("sender_account")["transaction_date"].diff().dt.days.fillna(0)
    )

    features = tx.merge(many_to_one, on="receiver_account", how="left")
    features = features.merge(one_to_many, on="sender_account", how="left")
    features = features.merge(mule_pair_total, on=["sender_account", "receiver_account"], how="left")

    for col in [
        "rule_many_to_one_count",
        "rule_one_to_many_count",
        "rule_mule_pair_total",
        "rule_dormant_gap_days",
    ]:
        features[col] = features[col].fillna(0)

    return features


def detect_many_to_one(transactions, min_unique_senders=5):
    grouped = transactions.groupby("receiver_account").agg({"sender_account": "nunique", "amount": "sum"})
    grouped = grouped[grouped["sender_account"] > min_unique_senders]
    return grouped.reset_index()


def detect_one_to_many(transactions, min_unique_receivers=5):
    grouped = transactions.groupby("sender_account").agg({"receiver_account": "nunique", "amount": "sum"})
    grouped = grouped[grouped["receiver_account"] > min_unique_receivers]
    return grouped.reset_index()


def detect_mule_accounts(transactions, min_pair_amount=10000):
    grouped = transactions.groupby(["sender_account", "receiver_account"]).agg({"amount": "sum"})
    grouped = grouped[grouped["amount"] > min_pair_amount]
    return grouped.reset_index()


def detect_dormant_accounts(accounts, transactions, dormant_cutoff="2025-01-01", active_from="2026-01-01"):
    dormant_accounts = accounts[accounts["last_active_date"] < dormant_cutoff]
    recent_transactions = transactions[transactions["transaction_date"] >= active_from]
    alerts = recent_transactions[recent_transactions["sender_account"].isin(dormant_accounts["account_id"])]
    return alerts.reset_index(drop=True)
