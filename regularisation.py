import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def _safe_float(value, default=1.0, minimum=0.0001):
    try:
        value = float(value)
        if value <= 0:
            return default
        return value
    except (ValueError, TypeError):
        return default


# ============================================================
# LINEAR REGRESSION + REGULARISATION
# Target: ARRIVAL_DELAY (continuous)
# ============================================================
def run_linear_regularisation(
    X_train,
    X_test,
    y_train,
    y_test,
    alpha=1.0,
    feature_names=None
):
    alpha = _safe_float(alpha)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Baseline Linear Regression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)

    # Ridge = L2
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    # Lasso = L1
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)

    # ElasticNet = L1 + L2
    elastic = ElasticNet(
        alpha=alpha,
        l1_ratio=0.5,
        max_iter=5000
    )
    elastic.fit(X_train, y_train)
    elastic_pred = elastic.predict(X_test)

    models = [
        ("Linear Regression", "None", linear, linear_pred),
        ("Ridge Regression", "L2", ridge, ridge_pred),
        ("Lasso Regression", "L1", lasso, lasso_pred),
        ("ElasticNet Regression", "L1 + L2", elastic, elastic_pred),
    ]

    results = []

    for name, penalty, model, pred in models:
        results.append({
            "model": name,
            "penalty": penalty,
            "alpha": round(alpha, 4),
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "r2": round(float(r2_score(y_test, pred)), 4),
            "active_features": int(
                np.sum(np.abs(np.asarray(model.coef_)) > 1e-4)
            ),
        })

    chart_path = None

    if feature_names is not None and len(feature_names) == len(linear.coef_):
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Linear Regression": linear.coef_,
            "Ridge (L2)": ridge.coef_,
            "Lasso (L1)": lasso.coef_,
            "ElasticNet": elastic.coef_,
        })

        melted = coef_df.melt(
            id_vars="Feature",
            var_name="Model",
            value_name="Weight"
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=melted,
            x="Feature",
            y="Weight",
            hue="Model"
        )
        plt.title(
            f"Linear Regression Regularisation - Alpha = {alpha}",
            fontsize=13
        )
        plt.xlabel("Features")
        plt.ylabel("Coefficient")
        plt.xticks(rotation=25)
        plt.tight_layout()

        file_path = os.path.join(
            CHARTS_DIR,
            "regularisation_linear_coeffs.png"
        )
        plt.savefig(file_path, dpi=110, bbox_inches="tight")
        plt.close()

        chart_path = "charts/regularisation_linear_coeffs.png"

    return {
        "alpha": alpha,
        "results": results,
        "chart_path": chart_path,
    }


# ============================================================
# LOGISTIC REGRESSION + REGULARISATION
# Target: IS_DELAYED (binary)
# ============================================================
def run_logistic_regularisation(
    X_train,
    X_test,
    y_train,
    y_test,
    C=1.0,
    feature_names=None
):
    C = _safe_float(C)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Logistic Regression without regularisation
    logistic_none = LogisticRegression(
        penalty=None,
        max_iter=3000,
        random_state=42
    )
    logistic_none.fit(X_train, y_train)
    none_pred = logistic_none.predict(X_test)
    none_prob = logistic_none.predict_proba(X_test)[:, 1]

    # L2
    logistic_l2 = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=3000,
        random_state=42
    )
    logistic_l2.fit(X_train, y_train)
    l2_pred = logistic_l2.predict(X_test)
    l2_prob = logistic_l2.predict_proba(X_test)[:, 1]

    # L1
    logistic_l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        max_iter=3000,
        random_state=42
    )
    logistic_l1.fit(X_train, y_train)
    l1_pred = logistic_l1.predict(X_test)
    l1_prob = logistic_l1.predict_proba(X_test)[:, 1]

    # ElasticNet
    logistic_elastic = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=C,
        max_iter=3000,
        random_state=42
    )
    logistic_elastic.fit(X_train, y_train)
    elastic_pred = logistic_elastic.predict(X_test)
    elastic_prob = logistic_elastic.predict_proba(X_test)[:, 1]

    models = [
        (
            "Logistic Regression",
            "None",
            logistic_none,
            none_pred,
            none_prob
        ),
        (
            "Logistic Regression",
            "L2",
            logistic_l2,
            l2_pred,
            l2_prob
        ),
        (
            "Logistic Regression",
            "L1",
            logistic_l1,
            l1_pred,
            l1_prob
        ),
        (
            "Logistic Regression",
            "ElasticNet (L1 + L2)",
            logistic_elastic,
            elastic_pred,
            elastic_prob
        ),
    ]

    results = []

    for name, penalty, model, pred, prob in models:
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = 0.0

        results.append({
            "model": name,
            "penalty": penalty,
            "C": round(C, 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(
                float(
                    precision_score(
                        y_test,
                        pred,
                        zero_division=0
                    )
                ),
                4
            ),
            "recall": round(
                float(
                    recall_score(
                        y_test,
                        pred,
                        zero_division=0
                    )
                ),
                4
            ),
            "f1": round(
                float(
                    f1_score(
                        y_test,
                        pred,
                        zero_division=0
                    )
                ),
                4
            ),
            "roc_auc": round(float(auc), 4),
            "active_features": int(
                np.sum(np.abs(model.coef_[0]) > 1e-4)
            ),
        })

    chart_path = None

    if feature_names is not None and len(feature_names) == len(logistic_none.coef_[0]):
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "No Regularisation": logistic_none.coef_[0],
            "L2": logistic_l2.coef_[0],
            "L1": logistic_l1.coef_[0],
            "ElasticNet": logistic_elastic.coef_[0],
        })

        melted = coef_df.melt(
            id_vars="Feature",
            var_name="Model",
            value_name="Weight"
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=melted,
            x="Feature",
            y="Weight",
            hue="Model"
        )
        plt.title(
            f"Logistic Regression Regularisation - C = {C}",
            fontsize=13
        )
        plt.xlabel("Features")
        plt.ylabel("Coefficient")
        plt.xticks(rotation=25)
        plt.tight_layout()

        file_path = os.path.join(
            CHARTS_DIR,
            "regularisation_logistic_coeffs.png"
        )
        plt.savefig(file_path, dpi=110, bbox_inches="tight")
        plt.close()

        chart_path = "charts/regularisation_logistic_coeffs.png"

    return {
        "C": C,
        "results": results,
        "chart_path": chart_path,
    }


# ============================================================
# MAIN FUNCTION
# Runs BOTH Linear and Logistic Regression regularisation
# ============================================================
def run_regularisation(
    X_train_reg,
    X_test_reg,
    y_train_reg,
    y_test_reg,
    X_train_clf,
    X_test_clf,
    y_train_clf,
    y_test_clf,
    alpha=1.0,
    C=1.0,
    feature_names=None
):
    linear_result = run_linear_regularisation(
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
        alpha=alpha,
        feature_names=feature_names
    )

    logistic_result = run_logistic_regularisation(
        X_train_clf,
        X_test_clf,
        y_train_clf,
        y_test_clf,
        C=C,
        feature_names=feature_names
    )

    return {
        "linear_regression": linear_result,
        "logistic_regression": logistic_result
    }


if __name__ == "__main__":
    from preprocessing import get_preprocessed_data

    (
        features,
        X_tr,
        X_te,
        yr_tr,
        yr_te,
        yc_tr,
        yc_te,
        _,
        _
    ) = get_preprocessed_data()

    output = run_regularisation(
        X_tr,
        X_te,
        yr_tr,
        yr_te,
        X_tr,
        X_te,
        yc_tr,
        yc_te,
        alpha=1.0,
        C=1.0,
        feature_names=features
    )

    print("\nLINEAR REGRESSION + REGULARISATION")
    for row in output["linear_regression"]["results"]:
        print(row)

    print("\nLOGISTIC REGRESSION + REGULARISATION")
    for row in output["logistic_regression"]["results"]:
        print(row)
