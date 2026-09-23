import os
import warnings
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

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


def run_eda(df):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    result = {}

    result["rows"] = int(df.shape[0])
    result["columns"] = int(df.shape[1])
    result["duplicate_rows"] = int(df.duplicated().sum())
    result["missing_values"] = (
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
        .sort_values(ascending=False)
        .to_dict()
    )
    result["columns_list"] = df.columns.tolist()
    result["charts"] = []

    sns.set_theme(style="whitegrid", font="sans-serif")
    accent_blue = "#0284c7"

    # 1. Flight Count by Airline
    if "AIRLINE" in df.columns:
        plt.figure(figsize=(9.5, 4.8))
        counts = df["AIRLINE"].value_counts().head(14)
        sns.barplot(x=counts.index, y=counts.values, color=accent_blue)
        plt.title("Flight Volume by Airline Carrier", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Airline Code", fontsize=11)
        plt.ylabel("Flight Count", fontsize=11)
        path = save_chart("1_airline_distribution.png")
        result["charts"].append({
            "title": "Flight Volume by Carrier",
            "desc": "Total scheduled flights across carriers in the sampled operational window.",
            "path": path,
            "category": "Traffic"
        })

    # 2. Average Arrival Delay by Airline
    if "AIRLINE" in df.columns and "ARRIVAL_DELAY" in df.columns:
        temp = df[["AIRLINE", "ARRIVAL_DELAY"]].dropna().copy()
        temp["ARRIVAL_DELAY"] = pd.to_numeric(temp["ARRIVAL_DELAY"], errors="coerce")
        airline_delay = temp.groupby("AIRLINE")["ARRIVAL_DELAY"].mean().sort_values(ascending=False)
        if not airline_delay.empty:
            plt.figure(figsize=(9.5, 5))
            colors = ["#ef4444" if v > 15 else "#0284c7" for v in airline_delay.values]
            sns.barplot(x=airline_delay.values, y=airline_delay.index, palette=colors)
            plt.axvline(15, color="#dc2626", linestyle="--", linewidth=1.5, label="15m FAA Delay Threshold")
            plt.title("Average Arrival Delay by Carrier (Minutes)", fontsize=13, fontweight="600", pad=10)
            plt.xlabel("Mean Arrival Delay (mins)", fontsize=11)
            plt.ylabel("Airline Code", fontsize=11)
            plt.legend(frameon=True)
            path = save_chart("2_average_delay_airline.png")
            result["charts"].append({
                "title": "Carrier Delay Comparison",
                "desc": "Mean delay per carrier. Bars exceeding 15 minutes trigger FAA delay classification.",
                "path": path,
                "category": "Punctuality"
            })

    # 3. Monthly Flight Volume
    if "MONTH" in df.columns:
        plt.figure(figsize=(9.5, 4.6))
        m_counts = df["MONTH"].value_counts().sort_index()
        month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        labels = [month_names.get(m, str(m)) for m in m_counts.index]
        sns.barplot(x=labels, y=m_counts.values, color="#38bdf8")
        plt.title("Monthly Scheduled Flight Frequency", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Month", fontsize=11)
        plt.ylabel("Number of Flights", fontsize=11)
        path = save_chart("3_flights_by_month.png")
        result["charts"].append({
            "title": "Monthly Flight Trends",
            "desc": "Monthly scheduled flight volume highlighting seasonal variation in demand.",
            "path": path,
            "category": "Seasonality"
        })

    # 4. Average Delay by Day of Week
    if "DAY_OF_WEEK" in df.columns and "ARRIVAL_DELAY" in df.columns:
        temp = df[["DAY_OF_WEEK", "ARRIVAL_DELAY"]].dropna().copy()
        temp["ARRIVAL_DELAY"] = pd.to_numeric(temp["ARRIVAL_DELAY"], errors="coerce")
        day_names = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
        day_avg = temp.groupby("DAY_OF_WEEK")["ARRIVAL_DELAY"].mean().reset_index()
        day_avg["Day"] = day_avg["DAY_OF_WEEK"].map(day_names)
        plt.figure(figsize=(9.5, 4.6))
        sns.barplot(x="Day", y="ARRIVAL_DELAY", data=day_avg, color="#0284c7")
        plt.title("Average Arrival Delay by Day of the Week", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Day of Week", fontsize=11)
        plt.ylabel("Average Arrival Delay (mins)", fontsize=11)
        path = save_chart("4_avg_delay_by_day_of_week.png")
        result["charts"].append({
            "title": "Weekday Delay Patterns",
            "desc": "Comparing weekday delays against weekend operational stability.",
            "path": path,
            "category": "Seasonality"
        })

    # 5. Daily Flights Distribution
    if "DAY" in df.columns:
        plt.figure(figsize=(10, 4.6))
        d_counts = df["DAY"].value_counts().sort_index()
        sns.barplot(x=d_counts.index, y=d_counts.values, color="#64748b")
        plt.title("Daily Flight Distribution (Day 1 to 31)", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Day of the Month", fontsize=11)
        plt.ylabel("Flight Count", fontsize=11)
        path = save_chart("5_flights_by_day.png")
        result["charts"].append({
            "title": "Daily Flight Distribution",
            "desc": "Flight density distributed across calendar days of the month.",
            "path": path,
            "category": "Traffic"
        })

    # 6. Top 10 Busiest Origin Airports
    if "ORIGIN_AIRPORT" in df.columns:
        plt.figure(figsize=(9.5, 4.8))
        top_orig = df["ORIGIN_AIRPORT"].astype(str).value_counts().head(10)
        sns.barplot(x=top_orig.index, y=top_orig.values, color="#0f766e")
        plt.title("Top 10 Origin Airports by Departures", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Origin Airport (IATA Code)", fontsize=11)
        plt.ylabel("Departing Flights", fontsize=11)
        path = save_chart("6_origin_airports.png")
        result["charts"].append({
            "title": "Top 10 Departure Hubs",
            "desc": "Busiest airport hubs originating flights across the domestic network.",
            "path": path,
            "category": "Airports"
        })

    # 7. Top 10 Busiest Destination Airports
    if "DESTINATION_AIRPORT" in df.columns:
        plt.figure(figsize=(9.5, 4.8))
        top_dest = df["DESTINATION_AIRPORT"].astype(str).value_counts().head(10)
        sns.barplot(x=top_dest.index, y=top_dest.values, color="#4338ca")
        plt.title("Top 10 Destination Airports by Arrivals", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Destination Airport (IATA Code)", fontsize=11)
        plt.ylabel("Arriving Flights", fontsize=11)
        path = save_chart("7_destination_airports.png")
        result["charts"].append({
            "title": "Top 10 Arrival Hubs",
            "desc": "High-density arrival terminals handling the largest passenger traffic.",
            "path": path,
            "category": "Airports"
        })

    # 8. Arrival Delay Distribution
    if "ARRIVAL_DELAY" in df.columns:
        plt.figure(figsize=(9.5, 4.8))
        arr_delay = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce").dropna()
        clipped_arr = arr_delay[(arr_delay >= -40) & (arr_delay <= 160)]
        sns.histplot(clipped_arr, bins=40, kde=True, color="#0284c7")
        plt.axvline(0, color="#16a34a", linestyle="--", label="On-Time (0m)")
        plt.axvline(15, color="#dc2626", linestyle="--", label="Delayed Threshold (15m)")
        plt.title("Arrival Delay Distribution (-40m to +160m)", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Arrival Delay (minutes)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.legend(frameon=True)
        path = save_chart("8_arrival_delay_distribution.png")
        result["charts"].append({
            "title": "Arrival Delay Distribution",
            "desc": "Histogram showing heavy positive skewness with majority flights arriving near on-time.",
            "path": path,
            "category": "Distribution"
        })

    # 9. Departure Delay Distribution
    if "DEPARTURE_DELAY" in df.columns:
        plt.figure(figsize=(9.5, 4.8))
        dep_delay = pd.to_numeric(df["DEPARTURE_DELAY"], errors="coerce").dropna()
        clipped_dep = dep_delay[(dep_delay >= -30) & (dep_delay <= 150)]
        sns.histplot(clipped_dep, bins=40, kde=True, color="#d97706")
        plt.axvline(0, color="#16a34a", linestyle="--", label="On-Time (0m)")
        plt.axvline(15, color="#dc2626", linestyle="--", label="Delayed (15m)")
        plt.title("Departure Delay Distribution (-30m to +150m)", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Departure Delay (minutes)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.legend(frameon=True)
        path = save_chart("9_departure_delay_distribution.png")
        result["charts"].append({
            "title": "Departure Delay Distribution",
            "desc": "Density curve of gate departure deviations across sampled flights.",
            "path": path,
            "category": "Distribution"
        })

    # 10. Delay Causes Breakdown
    delay_cause_cols = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
    present_causes = [c for c in delay_cause_cols if c in df.columns]
    if present_causes:
        cause_means = df[present_causes].mean().dropna().sort_values(ascending=False)
        if not cause_means.empty:
            plt.figure(figsize=(9.5, 4.8))
            clean_labels = [c.replace("_", " ").title() for c in cause_means.index]
            sns.barplot(x=clean_labels, y=cause_means.values, color="#e11d48")
            plt.title("Average Delay Duration by Delay Category", fontsize=13, fontweight="600", pad=10)
            plt.xlabel("Delay Reason", fontsize=11)
            plt.ylabel("Mean Duration (minutes)", fontsize=11)
            path = save_chart("10_delay_causes_breakdown.png")
            result["charts"].append({
                "title": "Delay Causes Breakdown",
                "desc": "Relative impact of late aircraft arrivals, air system congestion, and carrier delays.",
                "path": path,
                "category": "Operations"
            })

    # 11. HEATMAP 1: Correlation Heatmap
    corr_cols = [c for c in ["MONTH", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "TAXI_OUT", "SCHEDULED_TIME", "DISTANCE", "ARRIVAL_DELAY"] if c in df.columns]
    if len(corr_cols) >= 3:
        plt.figure(figsize=(9.5, 7.5))
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title("Correlation Matrix of Numerical Flight Attributes", fontsize=13, fontweight="600", pad=10)
        plt.xticks(rotation=35, ha="right")
        path = save_chart("11_correlation_heatmap.png")
        result["charts"].append({
            "title": "Feature Correlation Heatmap",
            "desc": "Pearson correlation coefficients highlighting strong coupling between departure and arrival delays.",
            "path": path,
            "category": "Heatmap"
        })

    # 12. HEATMAP 2: Day-of-Week vs Month Arrival Delay Heatmap
    if "MONTH" in df.columns and "DAY_OF_WEEK" in df.columns and "ARRIVAL_DELAY" in df.columns:
        temp = df[["MONTH", "DAY_OF_WEEK", "ARRIVAL_DELAY"]].dropna().copy()
        temp["ARRIVAL_DELAY"] = pd.to_numeric(temp["ARRIVAL_DELAY"], errors="coerce")
        day_names = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
        temp["Day"] = temp["DAY_OF_WEEK"].map(day_names)
        pivot = temp.pivot_table(index="Day", columns="MONTH", values="ARRIVAL_DELAY", aggfunc="mean")
        ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = pivot.reindex([d for d in ordered_days if d in pivot.index])
        plt.figure(figsize=(9.5, 5.5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Mean Delay (mins)'}, linewidths=0.5)
        plt.title("Average Arrival Delay: Day of Week vs. Month", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Month", fontsize=11)
        plt.ylabel("Day of Week", fontsize=11)
        path = save_chart("12_day_month_delay_heatmap.png")
        result["charts"].append({
            "title": "Seasonal Delay Matrix Heatmap",
            "desc": "Heatmap isolating historical high-delay intersection days and peak travel periods.",
            "path": path,
            "category": "Heatmap"
        })

    # 13. Departure Delay vs Arrival Delay Scatter
    if "DEPARTURE_DELAY" in df.columns and "ARRIVAL_DELAY" in df.columns:
        sample_df = df[["DEPARTURE_DELAY", "ARRIVAL_DELAY"]].dropna().sample(n=min(2500, len(df)), random_state=42).copy()
        sample_df = sample_df[(sample_df["DEPARTURE_DELAY"] >= -20) & (sample_df["DEPARTURE_DELAY"] <= 180) & (sample_df["ARRIVAL_DELAY"] >= -30) & (sample_df["ARRIVAL_DELAY"] <= 200)]
        plt.figure(figsize=(9.5, 5.2))
        sns.regplot(x="DEPARTURE_DELAY", y="ARRIVAL_DELAY", data=sample_df,
                    scatter_kws={'alpha': 0.25, 'color': '#0284c7', 's': 20},
                    line_kws={'color': '#dc2626', 'linewidth': 2, 'label': 'Regression Line'})
        plt.title("Departure Delay vs. Arrival Delay Fit", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Departure Delay (minutes)", fontsize=11)
        plt.ylabel("Arrival Delay (minutes)", fontsize=11)
        plt.legend(frameon=True)
        path = save_chart("13_dep_vs_arr_delay.png")
        result["charts"].append({
            "title": "Departure vs. Arrival Delay Fit",
            "desc": "Strong linear relationship showing departure delays directly propagate into arrival delays.",
            "path": path,
            "category": "Regression"
        })

    # 14. Distance vs Scheduled Flight Duration
    if "DISTANCE" in df.columns and "SCHEDULED_TIME" in df.columns:
        sample_dist = df[["DISTANCE", "SCHEDULED_TIME"]].dropna().sample(n=min(2000, len(df)), random_state=42).copy()
        plt.figure(figsize=(9.5, 5))
        sns.scatterplot(x="DISTANCE", y="SCHEDULED_TIME", data=sample_dist, color="#6366f1", alpha=0.35, s=20)
        plt.title("Scheduled Flight Duration vs. Flight Distance", fontsize=13, fontweight="600", pad=10)
        plt.xlabel("Distance (miles)", fontsize=11)
        plt.ylabel("Scheduled Time (minutes)", fontsize=11)
        path = save_chart("14_distance_vs_duration.png")
        result["charts"].append({
            "title": "Distance vs. Flight Duration",
            "desc": "Linear relationship between flight route mileage and scheduled block time.",
            "path": path,
            "category": "Regression"
        })

    return result


if __name__ == "__main__":
    from data_loading import load_dataset
    df = load_dataset()
    res = run_eda(df)
    print(f"Generated {len(res['charts'])} clean EDA plots.")
