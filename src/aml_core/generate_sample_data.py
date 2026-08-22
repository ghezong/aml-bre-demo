import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# Example usage:
# python aml_app/src/aml_core/generate_sample_data.py --num_transactions 50000 --data_type training --transactions_file training_rules.csv
# python aml_app/src/aml_core/generate_sample_data.py --num_transactions 10000 --data_type new --transactions_file new_rules.csv


def generate_transaction_data(num_records=1000, data_type="training"):
    random.seed(42)
    data = {
        "sender_account": [f"A{random.randint(1, 100)}" for _ in range(num_records)],
        "receiver_account": [f"B{random.randint(1, 100)}" for _ in range(num_records)],
        "amount": [random.uniform(10, 10000) for _ in range(num_records)],
        "transaction_date": [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(num_records)],
    }
    df = pd.DataFrame(data)
    if data_type == "training":
        df["alert"] = (df["amount"] > 5000).astype(int)
    return df


def generate_account_data(num_accounts=100):
    random.seed(42)
    data = {
        "account_id": [f"A{i}" for i in range(1, num_accounts + 1)],
        "last_active_date": [datetime.now() - timedelta(days=random.randint(365, 1000)) for _ in range(num_accounts)],
    }
    return pd.DataFrame(data)


def resolve_output_paths(project_root, data_type, transactions_file, accounts_file):
    data_dir = Path(project_root) / "aml_app" / "data"
    tx_dir = data_dir / ("training" if data_type == "training" else "new")
    acct_dir = data_dir / "accounts"
    tx_dir.mkdir(parents=True, exist_ok=True)
    acct_dir.mkdir(parents=True, exist_ok=True)
    return tx_dir / transactions_file, acct_dir / accounts_file


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AML sample data.")
    parser.add_argument("--num_transactions", type=int, default=1000, help="Number of transactions to generate")
    parser.add_argument("--transactions_file", type=str, default="sample_transactions.csv", help="Output file name for transactions")
    parser.add_argument("--num_accounts", type=int, default=100, help="Number of accounts to generate")
    parser.add_argument("--accounts_file", type=str, default="sample_accounts.csv", help="Output file name for accounts")
    parser.add_argument("--data_type", type=str, choices=["training", "new"], default="training", help="training includes alert label; new excludes labels")
    parser.add_argument("--project_root", type=str, default=None, help="Project root path. Defaults to repository root.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    project_root = Path(args.project_root) if args.project_root else default_root

    transactions = generate_transaction_data(args.num_transactions, args.data_type)
    accounts = generate_account_data(args.num_accounts)

    tx_path, acct_path = resolve_output_paths(project_root, args.data_type, args.transactions_file, args.accounts_file)

    transactions.to_csv(tx_path, index=False)
    accounts.to_csv(acct_path, index=False)

    print(f"Sample data generated and saved to {tx_path} and {acct_path}.")


if __name__ == "__main__":
    main()
