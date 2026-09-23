import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loading import load_dataset

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_CSV = os.path.join(BASE_DIR, "preprocessed_flights.csv")


def get_preprocessed_data():
    df = load_dataset()

    feature_cols = [
        'MONTH',
        'DAY',
        'DAY_OF_WEEK',
        'SCHEDULED_DEPARTURE',
        'DEPARTURE_DELAY',
        'DISTANCE',
        'SCHEDULED_TIME'
    ]
    target_col = 'ARRIVAL_DELAY'

    # Filter available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    needed_cols = available_cols + [target_col]

    # Drop nulls and extract clean copy
    df_clean = df[needed_cols].dropna().copy()

    # Numerical conversion
    for c in needed_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    df_clean = df_clean.dropna().copy()

    X_raw = df_clean[available_cols]
    y_reg = df_clean[target_col].values
    y_clf = (y_reg >= 15).astype(int)  # 1 = Delayed >= 15m, 0 = On-time

    # Add binary delay target to df_clean
    df_clean['IS_DELAYED'] = y_clf

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Attach scaled columns for easy inspection
    for idx, col in enumerate(available_cols):
        df_clean[f"{col}_SCALED"] = np.round(X_scaled[:, idx], 4)

    # Train / test split (80/20)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_reg, test_size=0.20, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X_scaled, y_clf, test_size=0.20, random_state=42
    )

    # Ensure preprocessed CSV is persisted
    if not os.path.exists(PREPROCESSED_CSV) or os.path.getsize(PREPROCESSED_CSV) == 0:
        df_clean.to_csv(PREPROCESSED_CSV, index=False)

    return available_cols, X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, df_clean, scaler


def generate_and_save_preprocessed_csv():
    """Explicitly regenerate preprocessed_flights.csv"""
    _, _, _, _, _, _, _, df_clean, _ = get_preprocessed_data()
    df_clean.to_csv(PREPROCESSED_CSV, index=False)
    return PREPROCESSED_CSV, len(df_clean)


if __name__ == "__main__":
    features, X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, df_c, sc = get_preprocessed_data()
    print("Preprocessing completed successfully!")
    print(f"Features ({len(features)}): {features}")
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")
    print(f"Delay Class distribution: On-time: {(yc_tr == 0).sum()}, Delayed: {(yc_tr == 1).sum()}")
