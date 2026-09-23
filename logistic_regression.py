import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def run_logistic_regression(X_train, X_test, y_train, y_test, feature_names=None):
    os.makedirs(CHARTS_DIR, exist_ok=True)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["On-Time (0)", "Delayed (1)"],
                yticklabels=["On-Time (0)", "Delayed (1)"])
    plt.title("Logistic Regression: Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual True Label", fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(CHARTS_DIR, "logistic_confusion_matrix.png")
    plt.savefig(cm_path, dpi=110, bbox_inches="tight")
    plt.close("all")

    # 2. Odds Ratios / Feature Coefficients
    feature_impacts = []
    if feature_names:
        odds_ratios = np.exp(model.coef_[0])
        for f, coef, odds in zip(feature_names, model.coef_[0], odds_ratios):
            feature_impacts.append({
                "feature": f,
                "coefficient": round(float(coef), 4),
                "odds_ratio": round(float(odds), 4)
            })
        feature_impacts.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

        plt.figure(figsize=(9, 5))
        f_names = [fi["feature"] for fi in feature_impacts]
        f_coefs = [fi["coefficient"] for fi in feature_impacts]
        colors = ["#08b6c9" if c >= 0 else "#f59e0b" for c in f_coefs]
        sns.barplot(x=f_coefs, y=f_names, palette=colors)
        plt.title("Logistic Regression: Log-Odds Feature Weights", fontsize=13, fontweight="bold")
        plt.xlabel("Coefficient (Log-Odds Impact)", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.axvline(0, color="#666", linestyle="--")
        plt.tight_layout()

        weights_path = os.path.join(CHARTS_DIR, "logistic_feature_weights.png")
        plt.savefig(weights_path, dpi=110, bbox_inches="tight")
        plt.close("all")

    return {
        "accuracy": f"{acc * 100:.2f}%",
        "precision": f"{prec * 100:.2f}%",
        "recall": f"{rec * 100:.2f}%",
        "f1": f"{f1 * 100:.2f}%",
        "roc_auc": round(float(roc_auc), 4),
        "total_test_samples": len(y_test),
        "true_positives": int(cm[1, 1]),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "cm_chart": "charts/logistic_confusion_matrix.png",
        "weights_chart": "charts/logistic_feature_weights.png",
        "feature_impacts": feature_impacts
    }


if __name__ == "__main__":
    from preprocessing import get_preprocessed_data
    features, X_tr, X_te, _, _, yc_tr, yc_te, _, _ = get_preprocessed_data()
    metrics = run_logistic_regression(X_tr, X_te, yc_tr, yc_te, feature_names=features)
    print("Logistic Regression Metrics:", metrics["accuracy"], "ROC-AUC:", metrics["roc_auc"])
