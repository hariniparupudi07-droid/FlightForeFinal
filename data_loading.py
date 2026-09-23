import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "flights.csv")
AIRLINES_PATH = os.path.join(BASE_DIR, "airlines.csv")
AIRPORTS_PATH = os.path.join(BASE_DIR, "airports.csv")

_CACHED_DF = None

def load_dataset(nrows=50000, reload=False):
    global _CACHED_DF
    if _CACHED_DF is None or reload:
        if os.path.exists(DATA_PATH):
            _CACHED_DF = pd.read_csv(DATA_PATH, low_memory=False, nrows=nrows)
        else:
            raise FileNotFoundError(f"flights.csv not found at {DATA_PATH}")
    return _CACHED_DF

def load_reference_data():
    airlines = {}
    airports = {}
    if os.path.exists(AIRLINES_PATH):
        try:
            df_air = pd.read_csv(AIRLINES_PATH)
            airlines = dict(zip(df_air['IATA_CODE'], df_air['AIRLINE']))
        except Exception:
            pass
    if os.path.exists(AIRPORTS_PATH):
        try:
            df_port = pd.read_csv(AIRPORTS_PATH)
            airports = dict(zip(df_port['IATA_CODE'], df_port['AIRPORT']))
        except Exception:
            pass
    return airlines, airports

def get_data_summary(df):
    airlines, airports = load_reference_data()

    first_5_rows = df.head(5).to_dict(orient="records")
    last_5_rows = df.tail(5).to_dict(orient="records")

    column_details = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        column_details.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].count()),
            "null_count": null_cnt,
            "null_pct": round((null_cnt / len(df)) * 100, 2)
        })

    numeric_cols = [c for c in ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE', 'SCHEDULED_TIME', 'AIR_TIME', 'TAXI_OUT'] if c in df.columns]
    desc = df[numeric_cols].describe().round(2).to_dict() if numeric_cols else {}

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "column_details": column_details,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "first_5_rows": first_5_rows,
        "last_5_rows": last_5_rows,
        "numeric_stats": desc,
        "numeric_cols": numeric_cols,
        "unique_airlines": int(df['AIRLINE'].nunique()) if 'AIRLINE' in df.columns else 0,
        "unique_origins": int(df['ORIGIN_AIRPORT'].nunique()) if 'ORIGIN_AIRPORT' in df.columns else 0,
        "unique_destinations": int(df['DESTINATION_AIRPORT'].nunique()) if 'DESTINATION_AIRPORT' in df.columns else 0,
        "airlines_lookup": airlines
    }
    return summary

if __name__ == "__main__":
    df = load_dataset()
    sum_data = get_data_summary(df)
    print(f"Loaded {sum_data['rows']} rows, {sum_data['columns']} cols, {sum_data['missing_values']} missing")
