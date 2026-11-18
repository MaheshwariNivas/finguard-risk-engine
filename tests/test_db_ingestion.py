import os
from decimal import Decimal
import pytest
import psycopg2
from psycopg2 import IntegrityError, OperationalError


@pytest.fixture(scope="module")
def db_conn():
    """Provides a module-scoped PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "finguard_db"),
            user=os.getenv("DB_USER", "rama"),
            password=os.getenv("DB_PASSWORD") or os.getenv("DB_PASS", "finguard123"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
        )
    except OperationalError as exc:
        pytest.fail(f"Database connection failed: {exc}")

    yield conn
    conn.close()


@pytest.fixture
def db_cursor(db_conn):
    """Provides a transactional cursor that rolls back after each test."""
    with db_conn.cursor() as cur:
        yield cur
    db_conn.rollback()


def test_row_parity_and_fraud_distribution(db_cursor):
    """Verifies exact row counts across normalized tables and fraud class parity."""
    db_cursor.execute("SELECT COUNT(*) FROM transactions;")
    tx_count = db_cursor.fetchone()[0]
    assert tx_count == 284_807, f"Expected 284,807 transactions, got {tx_count}"

    db_cursor.execute("SELECT COUNT(*) FROM transaction_features;")
    feat_count = db_cursor.fetchone()[0]
    assert feat_count == 284_807, f"Expected 284,807 feature rows, got {feat_count}"

    db_cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1;")
    fraud_count = db_cursor.fetchone()[0]
    assert fraud_count == 492, f"Expected 492 fraud cases, got {fraud_count}"


def test_exact_monetary_volume(db_cursor):
    """Ensures NUMERIC(12, 2) avoided IEEE-754 floating-point drift."""
    db_cursor.execute("SELECT SUM(amount) FROM transactions;")
    total_volume = db_cursor.fetchone()[0]
    expected_volume = Decimal("25162590.01")

    assert isinstance(total_volume, Decimal), "SUM(amount) must return Decimal"
    assert total_volume == expected_volume, (
        f"Monetary drift detected: expected {expected_volume}, got {total_volume}"
    )


def test_referential_integrity_and_no_orphans(db_cursor):
    """Confirms 1:1 relational mapping with zero orphaned child or parent records."""
    db_cursor.execute(
        """
        SELECT COUNT(*)
        FROM transactions t
        FULL OUTER JOIN transaction_features tf
            ON t.transaction_id = tf.transaction_id
        WHERE t.transaction_id IS NULL OR tf.transaction_id IS NULL;
        """
    )
    orphan_count = db_cursor.fetchone()[0]
    assert orphan_count == 0, f"Found {orphan_count} unjoined/orphaned rows"


def test_foreign_key_and_domain_constraints(db_cursor):
    """Enforces database-level rejection of orphan features and invalid classes."""
    # 1. Reject orphaned child insertion
    with pytest.raises(IntegrityError):
        db_cursor.execute(
            """
            INSERT INTO transaction_features (transaction_id, v1)
            VALUES (-999999, 0.0);
            """
        )
    db_cursor.connection.rollback()

    # 2. Reject invalid binary target class
    with pytest.raises(IntegrityError):
        db_cursor.execute(
            """
            INSERT INTO transactions (time_elapsed_sec, amount, is_fraud)
            VALUES (172800, 10.00, 2);
            """
        )
    db_cursor.connection.rollback()


def test_serial_sequence_alignment(db_cursor):
    """Verifies setval resynchronization prevents primary key collision on next insert."""
    db_cursor.execute("SELECT MAX(transaction_id) FROM transactions;")
    max_id = db_cursor.fetchone()[0]

    db_cursor.execute(
        "SELECT last_value, is_called FROM transactions_transaction_id_seq;"
    )
    last_val, is_called = db_cursor.fetchone()
    next_expected = last_val + 1 if is_called else last_val

    assert next_expected > max_id, (
        f"Sequence out of sync: next ID {next_expected} <= max ID {max_id}"
    )

    # Prove live insert succeeds and rolls back cleanly
    db_cursor.execute(
        """
        INSERT INTO transactions (time_elapsed_sec, amount, is_fraud)
        VALUES (172801, 1.00, 0)
        RETURNING transaction_id;
        """
    )
    inserted_id = db_cursor.fetchone()[0]
    assert inserted_id > max_id

