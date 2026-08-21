"""
Phase 1 – Exploratory Data Analysis
Run this script to generate key statistics, plots (saved to notebooks/eda_plots/),
and a concise findings report.

Usage:
    python notebooks/01_eda.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.load import load_raw_data
from src.config import (
    TARGET_COL,
    TARGET_LABEL_COL,
    DATETIME_COL,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
)

# Output directory for plots
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


def main():
    print("=" * 70)
    print("FRAUD DETECTION – PHASE 1 EDA")
    print("=" * 70)

    df = load_raw_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range   : {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}")
    print(f"Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # ------------------------------------------------------------------
    # 1. Target distribution
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. TARGET DISTRIBUTION")
    print("-" * 50)
    fraud_rate = df[TARGET_COL].mean()
    print(df[TARGET_LABEL_COL].value_counts())
    print(f"\nFraud rate: {fraud_rate:.2%}")
    print(f"Imbalance ratio (legit:fraud) ≈ {(1 - fraud_rate) / fraud_rate:.1f}:1")

    fig, ax = plt.subplots()
    df[TARGET_LABEL_COL].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Fraud vs Legitimate Transactions")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # 2. Temporal patterns
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. TEMPORAL PATTERNS")
    print("-" * 50)

    df["YearMonth"] = df[DATETIME_COL].dt.to_period("M")
    monthly = df.groupby("YearMonth")[TARGET_COL].agg(["count", "mean"])
    print("\nMonthly volume & fraud rate (first 6 / last 6 months):")
    print(monthly.head(6))
    print("...")
    print(monthly.tail(6))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    monthly["count"].plot(ax=axes[0], title="Transaction Volume by Month")
    axes[0].set_ylabel("Count")
    monthly["mean"].plot(ax=axes[1], title="Fraud Rate by Month", color="#e74c3c")
    axes[1].set_ylabel("Fraud Rate")
    axes[1].axhline(fraud_rate, color="gray", linestyle="--", label="Overall average")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_temporal_trends.png", dpi=120)
    plt.close()

    # Hour / Day of week
    print("\nFraud rate by hour of day:")
    print(df.groupby("Transaction_Hour")[TARGET_COL].mean().round(3))

    print("\nFraud rate by day of week (0=Mon ... 6=Sun):")
    print(df.groupby("Day_of_Week")[TARGET_COL].mean().round(3))

    # ------------------------------------------------------------------
    # 3. Numeric feature distributions vs target
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. NUMERIC FEATURES – SUMMARY BY TARGET")
    print("-" * 50)

    summary = df.groupby(TARGET_COL)[NUMERIC_FEATURES].agg(["mean", "median", "std"])
    print(summary.T.round(2).to_string())

    # High-signal risk scores
    risk_cols = ["IP_Risk_Score", "Velocity_Score", "Merchant_Risk_Score"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, risk_cols):
        sns.boxplot(data=df, x=TARGET_LABEL_COL, y=col, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_risk_scores_by_target.png", dpi=120)
    plt.close()

    # Amount distributions (log scale useful)
    fig, ax = plt.subplots()
    for label, color in [("Legitimate", "#2ecc71"), ("Fraudulent", "#e74c3c")]:
        subset = df[df[TARGET_LABEL_COL] == label]["Transaction_Amount"]
        ax.hist(np.log1p(subset), bins=50, alpha=0.6, label=label, color=color)
    ax.set_xlabel("log1p(Transaction_Amount)")
    ax.set_title("Transaction Amount Distribution (log scale)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_amount_distribution.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # 4. Binary flags – fraud lift
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("4. BINARY FLAGS – FRAUD RATE & LIFT")
    print("-" * 50)

    overall = df[TARGET_COL].mean()
    for col in BINARY_FEATURES:
        rates = df.groupby(col)[TARGET_COL].mean()
        print(f"\n{col}:")
        for val, rate in rates.items():
            lift = rate / overall if overall > 0 else np.nan
            print(f"  value={val}  fraud_rate={rate:.2%}  lift={lift:.2f}x")

    # ------------------------------------------------------------------
    # 5. Categorical features – fraud rate
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("5. CATEGORICAL FEATURES – FRAUD RATE")
    print("-" * 50)

    for col in CATEGORICAL_FEATURES:
        rates = (
            df.groupby(col)[TARGET_COL]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
        )
        rates["mean"] = rates["mean"].map(lambda x: f"{x:.2%}")
        print(f"\n{col}:")
        print(rates.to_string())

    # ------------------------------------------------------------------
    # 6. Correlation (numeric only)
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("6. CORRELATION WITH TARGET (numeric)")
    print("-" * 50)

    corr = df[NUMERIC_FEATURES + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(corr_sorted.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 10))
    corr_sorted.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Pearson Correlation with Fraud_Flag")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_correlation_with_target.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # 7. Missing values & duplicates
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("7. DATA QUALITY")
    print("-" * 50)
    print(f"Missing values total : {df.isnull().sum().sum()}")
    print(f"Duplicate rows       : {df.duplicated().sum()}")
    print(f"Duplicate IDs        : {df['Transaction_ID'].duplicated().sum()}")

    # ------------------------------------------------------------------
    # 8. Key findings summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY (Phase 1)")
    print("=" * 70)
    print(
        """
• Fraud rate is ~24.9% – moderately imbalanced. Prefer PR-AUC / F1 / cost-sensitive
  metrics over plain accuracy.

• Strong individual signals already present:
  - IP_Risk_Score, Velocity_Score, Merchant_Risk_Score
  - Previous_Chargebacks, Failed_Payment_Attempts, Login_Anomalies
  - Shipping_Billing_Mismatch, New_Device, VPN_Proxy_Used, High_Risk_Country

• Temporal structure exists (multi-year data). Use time-aware splits to avoid leakage.

• Transaction_Amount has a long right tail; log-transform or robust scaling may help
  linear models (trees are less sensitive).

• No missing values and unique Transaction_IDs → clean starting point.

• Business metric recommendation:
  Optimize a cost-sensitive threshold or maximize Precision-Recall AUC,
  then tune operating point according to Cost_FP ≈ 5 vs Cost_FN ≈ 100
  (adjust these numbers with real finance/ops input).
"""
    )
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("Phase 1 EDA complete.")


if __name__ == "__main__":
    main()