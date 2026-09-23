import os
from preprocessing import generate_and_save_preprocessed_csv

if __name__ == "__main__":
    print("Generating preprocessed_flights.csv...")
    csv_path, rows = generate_and_save_preprocessed_csv()
    print(f"SUCCESS! Saved {rows} cleaned records to: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")
