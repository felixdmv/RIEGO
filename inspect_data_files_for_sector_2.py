import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
canduela_dir = os.path.join(base_path, "datos", "canduela")

# Search all CSV files in canduela_dir for "sector" or "superficie" or "caudal"
for f in os.listdir(canduela_dir):
    if f.endswith(".csv"):
        path = os.path.join(canduela_dir, f)
        try:
            df = pd.read_csv(path, nrows=5)
            cols = list(df.columns)
            # Check if any column contains "sector" or "caudal" or "superficie"
            matching_cols = [c for c in cols if any(k in c.lower() for k in ["sector", "caudal", "superficie", "area", "flow"])]
            if matching_cols:
                print(f"File: {f} has matching columns: {matching_cols}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

# Check unique values in columns of CSVs in canduela_dir
print("\n--- CHECKING VALUES IN CSVs ---")
csvs = ["RIEGOS_1.csv", "Dataset_Final_S3.csv", "Dataset_Multimodal_S3_Tesis.csv", "Master_Sector3_Audit.csv"]
for name in csvs:
    path = os.path.join(canduela_dir, name)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"\nFile: {name}, Shape: {df.shape}")
            for col in df.columns:
                if 'sector' in col.lower() or 'válvula' in col.lower() or 'valvula' in col.lower() or 'nombre' in col.lower():
                    print(f"  Col '{col}' unique values: {df[col].dropna().unique()[:10]}")
        except Exception as e:
            print(f"Error reading {name}: {e}")
