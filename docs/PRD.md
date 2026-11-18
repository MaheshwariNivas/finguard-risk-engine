# Product Requirements Document (PRD): FinGuard AI
**Owner:** Maheshwari Nivas | **Status:** Active (Week 1) | **Target SLA:** p99 < 50ms

## 1. Problem Statement & Business Context
Payment processors face an asymmetric risk problem: fraudulent transactions represent only **0.1727%** of volume (492 out of 284,807 transactions totaling $25.16M), yet unflagged fraud causes direct financial chargebacks and network penalties. Conversely, aggressive blocking creates false positives that frustrate legitimate cardholders and cause customer churn.

## 2. Success Metrics (Business & Technical KPIs)
* **Primary Business Metric (Net Financial Loss Minimized):** Optimize decision threshold $\tau^*$ using an explicit cost matrix rather than a default $0.50$ probability cutoff:
  * **False Negative Cost ($C_{FN}$):** Full transaction amount ($\text{Amount}_i$) lost to chargeback.
  * **False Positive Cost ($C_{FP}$):** Fixed operational & customer friction cost ($10.00 per blocked legitimate transaction).
* **Primary ML Metric:** Precision-Recall AUC (**PR-AUC**) and Brier Score (probability calibration reliability). ROC-AUC is explicitly deprioritized due to severe class imbalance.
* **System Latency SLA:** FastAPI real-time scoring endpoint must achieve **p99 latency < 50ms** under concurrent load.

## 3. Scope & Delivery Milestones
* **Milestone 1 (Data & Storage Foundation):** Normalized PostgreSQL schema (`transactions`, `transaction_features`), high-speed idempotent `COPY` ingestion, and automated `pytest` data quality suite.
* **Milestone 2 (Leakage-Free ML & Cost Optimization):** Strict temporal train/validation/test split, XGBoost classifier with Platt/Isotonic probability calibration, cost-sensitive thresholding, and SHAP explainability.
* **Milestone 3 (Production Inference & Monitoring):** Low-latency FastAPI microservice with p99 benchmarking, SQL drift/velocity queries, and an interactive Streamlit risk operations dashboard.
