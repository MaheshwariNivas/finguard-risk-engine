# ADR-001: Normalized PostgreSQL Schema & Precision Storage

**Status:** Accepted  
**Context:** Storing 284,807 financial transactions (totaling $25,162,590.01) alongside 28 anonymized PCA feature vectors (`V1`–`V28`) for both analytical queries and ML model training.

## Decisions
1. **1-to-1 Table Normalization (`transactions` vs. `transaction_features`):**
   * Separated core transactional metadata (`transaction_id`, `time_elapsed_sec`, `amount`, `is_fraud`, `created_at`) from the 28-column feature vector.
   * **Trade-off:** Requires a primary-key `JOIN` when assembling ML training matrices, which adds minimal overhead (<150ms) while keeping financial/operational queries lightweight and decoupled from model feature evolution.
2. **Exact Fixed-Point Financial Precision (`NUMERIC(12, 2)`):**
   * Used `NUMERIC(12, 2)` instead of `DOUBLE PRECISION` for `amount` to eliminate IEEE-754 floating-point rounding drift during aggregate loss and volume calculations.
3. **Targeted Indexing Strategy:**
   * Created `idx_transactions_time` on `time_elapsed_sec` to enforce fast chronological slicing and prevent temporal data leakage during train/test splitting.
   * Created a **partial B-Tree index** (`WHERE is_fraud = 1`) on `is_fraud`, reducing index footprint by >99.8% while accelerating minority-class lookups.
4. **Idempotent Bulk Ingestion via `COPY` Protocol:**
   * Used PostgreSQL `COPY FROM STDIN` with explicit sequence resynchronization (`setval`) so batch reloads execute atomically without breaking future API `SERIAL` inserts.
