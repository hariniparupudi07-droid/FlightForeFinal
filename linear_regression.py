import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def run_linear_regression(X_train, X_test, y_train, y_test, feature_names=None, selected_feature=None, df_clean=None):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    metrics = {
        "selected_feature": selected_feature or "ALL",
        "available_features": feature_names or [],
        "is_single_feature": False
    }

    # Case 1: Single Feature selected
    if selected_feature and selected_feature != "ALL" and feature_names and selected_feature in feature_names:
        feat_idx = feature_names.index(selected_feature)
        X_tr_feat = X_train[:, feat_idx:feat_idx+1]
        X_te_feat = X_test[:, feat_idx:feat_idx+1]

        model = LinearRegression()
        model.fit(X_tr_feat, y_train)
        y_pred = model.predict(X_te_feat)

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics.update({
            "is_single_feature": True,
            "feature_name": selected_feature,
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "equation": f"ARRIVAL_DELAY = {slope:.4f} * ({selected_feature}) + ({intercept:.4f})",
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4)
        })

        # Generate Regression Fit Plot
        plt.figure(figsize=(9, 5))
        # Use a sub-sample for fast & clean plotting
        n_pts = min(1500, len(X_te_feat))
        sample_indices = np.random.RandomState(42).choice(len(X_te_feat), n_pts, replace=False)
        x_sub = X_te_feat[sample_indices, 0]
        y_sub = y_test[sample_indices]

        plt.scatter(x_sub, y_sub, alpha=0.3, color="#08b6c9", label="Actual Test Points")
        # Line points
        x_line = np.linspace(x_sub.min(), x_sub.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        plt.plot(x_line, y_line, color="#dc2626", linewidth=2.5, label="Fitted Regression Line")

        plt.title(f"Linear Fit: {selected_feature} vs. ARRIVAL_DELAY (R² = {r2:.4f})", fontsize=13, fontweight="bold")
        plt.xlabel(f"{selected_feature} (Standardized)", fontsize=11)
        plt.ylabel("Arrival Delay (minutes)", fontsize=11)
        plt.legend()
        plt.tight_layout()

        chart_filename = f"linear_reg_{selected_feature.lower()}.png"
        chart_path = os.path.join(CHARTS_DIR, chart_filename)
        plt.savefig(chart_path, dpi=110, bbox_inches="tight")
        plt.close("all")
        metrics["chart_path"] = f"charts/{chart_filename}"

    # Case 2: All features (Multiple Linear Regression)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        coeffs = []
        if feature_names:
            for feat, coef in zip(feature_names, model.coef_):
                coeffs.append({"feature": feat, "coef": round(float(coef), 4)})
            coeffs.sort(key=lambda x: abs(x["coef"]), reverse=True)

        metrics.update({
            "is_single_feature": False,
            "feature_name": "All Features",
            "intercept": round(float(model.intercept_), 4),
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "coefficients": coeffs
        })

        # Generate Coefficients Bar Chart
        if coeffs:
            plt.figure(figsize=(9, 5))
            f_names = [c["feature"] for c in coeffs]
            f_vals = [c["coef"] for c in coeffs]
            colors = ["#08b6c9" if v >= 0 else "#f59e0b" for v in f_vals]
            sns.barplot(x=f_vals, y=f_names, palette=colors)
            plt.title("Multiple Linear Regression: Feature Coefficients (Weights)", fontsize=13, fontweight="bold")
            plt.xlabel("Coefficient Value (Standardized Impact)", fontsize=11)
            plt.ylabel("Feature", fontsize=11)
            plt.axvline(0, color="#666", linestyle="--")
            plt.tight_layout()

            chart_filename = "linear_reg_all_coeffs.png"
            chart_path = os.path.join(CHARTS_DIR, chart_filename)
            plt.savefig(chart_path, dpi=110, bbox_inches="tight")
            plt.close("all")
            metrics["chart_path"] = f"charts/{chart_filename}"

    return metrics


if __name__ == "__main__":
    from preprocessing import get_preprocessed_data
    features, X_tr, X_te, yr_tr, yr_te, _, _, df_c, _ = get_preprocessed_data()
    m_all = run_linear_regression(X_tr, X_te, yr_tr, yr_te, feature_names=features)
    print("Linear Regression (All Features):", m_all["r2"], m_all["mae"])
    m_single = run_linear_regression(X_tr, X_te, yr_tr, yr_te, feature_names=features, selected_feature="DEPARTURE_DELAY")
    print("Linear Regression (DEPARTURE_DELAY):", m_single["r2"], m_single["mae"])
