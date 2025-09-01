
#!/usr/bin/env python3
"""dma818_pipeline.py — reusable ML pipeline for Boston Housing.
Runs EDA + regression + classification, saves metrics and charts.

Usage:
  python dma818_pipeline.py --data BostonHousing.csv --out outdir

Outputs:
  - metrics JSON
  - charts (PNG)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="BostonHousing.csv", help="Path to CSV dataset")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["MEDV", "CAT. MEDV"])
    y_reg = df["MEDV"]
    y_clf = df["CAT. MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=args.test_size, random_state=args.seed)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=args.test_size, random_state=args.seed)

    # Regression
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Classification
    clf = LogisticRegression(max_iter=1000).fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    acc = float(accuracy_score(y_test_c, y_pred_c))
    report = classification_report(y_test_c, y_pred_c, output_dict=True)
    cm = confusion_matrix(y_test_c, y_pred_c)

    # Metrics JSON
    metrics = {
        "regression": {"rmse": rmse, "r2": r2},
        "classification": {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Charts (matplotlib only)
    # Heatmap
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_title("Correlation Heatmap (Numeric)")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    save_fig(fig, outdir / "corr_heatmap.png")

    # MEDV hist
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["MEDV"].values, bins=30)
    ax.set_title("Distribution of MEDV"); ax.set_xlabel("MEDV"); ax.set_ylabel("Frequency")
    save_fig(fig, outdir / "medv_hist.png")

    # RM vs MEDV
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df["RM"].values, df["MEDV"].values, s=10)
    ax.set_title("RM vs MEDV"); ax.set_xlabel("RM"); ax.set_ylabel("MEDV")
    save_fig(fig, outdir / "rm_vs_medv.png")

    # LSTAT vs MEDV
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df["LSTAT"].values, df["MEDV"].values, s=10)
    ax.set_title("LSTAT vs MEDV"); ax.set_xlabel("LSTAT"); ax.set_ylabel("MEDV")
    save_fig(fig, outdir / "lstat_vs_medv.png")

    # Residuals
    fig, ax = plt.subplots(figsize=(6,4))
    resid = y_test.values - y_pred
    ax.scatter(y_pred, resid, s=12); ax.axhline(0, linestyle="--")
    ax.set_title("Residuals vs Predicted"); ax.set_xlabel("Predicted MEDV"); ax.set_ylabel("Residual")
    save_fig(fig, outdir / "residuals.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    save_fig(fig, outdir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
