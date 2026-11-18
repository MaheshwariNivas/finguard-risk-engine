import io
import logging
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/creditcard.csv")

# Connects via local Unix domain socket (/var/run/postgresql) using peer auth by default.
# Supports TCP/IP via PGHOST & PGPASSWORD when deployed in Docker/Production.
DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "finguard_db"),
    "user": os.getenv("PGUSER", "rama"),
}
if os.getenv("PGHOST"):
    DB_CONFIG["host"] = os.getenv("PGHOST")
if os.getenv("PGPORT"):
    DB_CONFIG["port"] = os.getenv("PGPORT")
if os.getenv("PGPASSWORD"):
    DB_CONFIG["password"] = os.getenv("PGPASSWORD")


def validate_dataframe(df: pd.DataFrame) -> None:
    """Enforce strict data quality invariants before database ingestion."""
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if df.isnull().any().any():
        raise ValueError("Null values detected in raw dataset; aborting ingestion.")

    if (df["Time"] < 0).any() or (df["Amount"] < 0).any():
        raise ValueError("Negative Time or Amount values violate domain constraints.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Invalid target labels found in Class column.")


def dataframe_to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """Serialize DataFrame to an in-memory CSV buffer for PostgreSQL COPY."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    return buffer


def ingest_data(csv_path: Path = DATA_PATH) -> None:
    """Load, validate, normalize, and bulk-insert creditcard.csv via COPY."""
    start_time = time.perf_counter()

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")

    logger.info("Reading raw dataset from %s...", csv_path)
    df = pd.read_csv(csv_path)
    validate_dataframe(df)

    df = df.reset_index(drop=True)
    df["transaction_id"] = df.index + 1

    # Vectorized rounding (100x faster than Python .map lambda)
    transactions_df = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "time_elapsed_sec": df["Time"].astype("int64"),
        "amount": df["Amount"].round(2),
        "is_fraud": df["Class"].astype("int16"),
    })

    feature_cols = [f"V{i}" for i in range(1, 29)]
    features_df = df[["transaction_id"] + feature_cols]

    tx_buffer = dataframe_to_csv_buffer(transactions_df)
    feat_buffer = dataframe_to_csv_buffer(features_df)

    logger.info("Connecting to PostgreSQL (%s)...", DB_CONFIG["dbname"])
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE transaction_features, transactions RESTART IDENTITY CASCADE;")

                cur.copy_expert(
                    """
                    COPY transactions (transaction_id, time_elapsed_sec, amount, is_fraud)
                    FROM STDIN WITH (FORMAT CSV)
                    """,
                    tx_buffer,
                )

                feat_col_sql = ", ".join(["transaction_id"] + [c.lower() for c in feature_cols])
                cur.copy_expert(
                    f"""
                    COPY transaction_features ({feat_col_sql})
                    FROM STDIN WITH (FORMAT CSV)
                    """,
                    feat_buffer,
                )

                cur.execute(
                    "SELECT setval(pg_get_serial_sequence('transactions', 'transaction_id'), MAX(transaction_id)) FROM transactions;"
                )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Successfully ingested %d transactions in %.2f seconds (%.0f rows/sec).",
            len(df),
            elapsed,
            len(df) / elapsed,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    ingest_data()


