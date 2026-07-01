import pandas as pd
import os
import glob

base_path = "datos/canduela"

file_mini1 = os.path.join(base_path, "CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx")
file_mini2 = os.path.join(base_path, "florapulse", "CANDUELA MINI 2 ALMENDRO 2025.xlsx")

for name, path in [("MINI 1 (2026)", file_mini1), ("MINI 2 (2025)", file_mini2)]:
    print(f"\n=== {name} ===")
    if os.path.exists(path):
        xl = pd.ExcelFile(path)
        print("Sheets:", xl.sheet_names)
        df = xl.parse(xl.sheet_names[0])
        print("Columns:", list(df.columns))
        print("Shape:", df.shape)
        # Find date column
        fecha_col = [c for c in df.columns if 'fecha' in str(c).lower()][0]
        df[fecha_col] = pd.to_datetime(df[fecha_col])
        print("Min date:", df[fecha_col].min())
        print("Max date:", df[fecha_col].max())
    else:
        print("File not found.")
