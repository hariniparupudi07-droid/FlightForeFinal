import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from preprocessing import get_preprocessed_data

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def save_chart(filename):
    path = chart_path(filename)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return "charts/" + filename


def load_flight_clustering_data(sample_size=4000):
    """Extract and standardize flight features suitable for clustering."""
    _, _, _, _, _, _, _, df_clean, _ = get_preprocessed_data()

    features = ['DISTANCE', 'SCHEDULED_TIME', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']
    available = [f for f in features if f in df_clean.columns]

    df_sub = df_clean[available].dropna().copy()
    if len(df_sub) > sample_size:
        df_sub = df_sub.sample(n=sample_size, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sub)

    return df_sub, X_scaled, available


def run_elbow_method(X, k_range=range(2, 11)):
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(X)
        wcss.append(round(float(kmeans.inertia_), 2))

    plt.figure(figsize=(9, 4.8))
    plt.plot(list(k_range), wcss, marker="o", color="#08b6c9", linewidth=2, markersize=7)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("WCSS Score (Within-Cluster Sum of Squares)", fontsize=11)
    plt.title("Elbow Method For Optimal K (Flight Clusters)", fontsize=13, fontweight="bold")
    plt.xticks(list(k_range))
    plt.grid(True, linestyle="--", alpha=0.6)
    path = save_chart("elbow_method.png")

    return {
        "k_values": list(k_range),
        "wcss": wcss,
        "chart": path
    }


def run_silhouette_method(X, k_range=range(2, 11)):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(2500, len(X)), random_state=42)
        scores.append(round(float(score), 4))

    best_index = int(np.argmax(scores))
    best_k = list(k_range)[best_index]

    plt.figure(figsize=(9, 4.8))
    plt.plot(list(k_range), scores, marker="s", color="#6366f1", linewidth=2, markersize=7)
    plt.axvline(best_k, color="#dc2626", linestyle="--", label=f"Optimal K = {best_k}")
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Silhouette Score", fontsize=11)
    plt.title("Silhouette Score Method for K-Means", fontsize=13, fontweight="bold")
    plt.xticks(list(k_range))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    path = save_chart("silhouette_method.png")

    return {
        "k_values": list(k_range),
        "scores": scores,
        "best_k": best_k,
        "best_score": max(scores),
        "chart": path
    }


def run_kmeans_clustering(df_sub, X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {f"Cluster {int(u)}": int(c) for u, c in zip(unique, counts)}

    # PCA 2D Cluster Visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(9, 5.5))
    palette = sns.color_palette("tab10", k)
    for c_id in range(k):
        mask = labels == c_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c_id}", color=palette[c_id], alpha=0.5, s=25)

    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], color="#111", marker="*", s=220, edgecolors="white", linewidths=1.5, label="Centroids")
    plt.title(f"K-Means Cluster Distribution (K={k}, PCA Projection)", fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    scatter_chart = save_chart("kmeans_clusters.png")

    # Cluster profiling table
    df_prof = df_sub.copy()
    df_prof["Cluster"] = labels
    profiles = []
    for c_id in range(k):
        sub = df_prof[df_prof["Cluster"] == c_id]
        profiles.append({
            "cluster_id": c_id,
            "count": len(sub),
            "percentage": f"{(len(sub) / len(df_prof) * 100):.1f}%",
            "avg_distance": round(float(sub["DISTANCE"].mean()), 1) if "DISTANCE" in sub else "-",
            "avg_sched_time": round(float(sub["SCHEDULED_TIME"].mean()), 1) if "SCHEDULED_TIME" in sub else "-",
            "avg_dep_delay": round(float(sub["DEPARTURE_DELAY"].mean()), 1) if "DEPARTURE_DELAY" in sub else "-",
            "avg_arr_delay": round(float(sub["ARRIVAL_DELAY"].mean()), 1) if "ARRIVAL_DELAY" in sub else "-"
        })

    return {
        "k": k,
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_sizes": cluster_sizes,
        "cluster_chart": scatter_chart,
        "profiles": profiles,
        "labels_preview": labels[:15].tolist()
    }


def run_kmeans(method="silhouette", manual_k=None):
    """Master entry point used by the Flask route."""
    df_sub, X, _ = load_flight_clustering_data(sample_size=3500)
    result = {"method": method}

    if method == "elbow":
        result["elbow"] = run_elbow_method(X)
        result["clustering"] = None
        result["note"] = "Inspect the WCSS curve above, identify the inflection bend (elbow), then choose a manual K."
        return result

    elif method == "silhouette":
        sil = run_silhouette_method(X)
        result["silhouette"] = sil
        k = sil["best_k"]

    elif method == "manual":
        if manual_k is None or str(manual_k).strip() == "" or int(manual_k) < 2:
            result["error"] = "Please provide a valid number of clusters K (>= 2)."
            return result
        k = min(10, max(2, int(manual_k)))

    else:
        result["error"] = "Invalid method selected."
        return result

    result["clustering"] = run_kmeans_clustering(df_sub, X, k)
    return result


if __name__ == "__main__":
    print("Testing Flight KMeans Clustering...")
    res = run_kmeans(method="silhouette")
    print("Silhouette method chosen K:", res["silhouette"]["best_k"])
    print("Cluster sizes:", res["clustering"]["cluster_sizes"])
