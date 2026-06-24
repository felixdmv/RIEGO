import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
fp_dir = os.path.join(base_path, "datos", "canduela", "florapulse")

keywords = ["caudal", "superficie", "area", "m2", "ha", "litro", "m3/h", "l/h", "diseño"]

print("--- SCANNING FLORAPULSE EXCEL FILES ---")
for f in os.listdir(fp_dir):
    if f.endswith(".xlsx"):
        path = os.path.join(fp_dir, f)
        print(f"\nFile: {f}")
        try:
            xl = pd.ExcelFile(path)
            print("  Sheets:", xl.sheet_names)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet, nrows=20) # check first 20 rows of each sheet
                for r_idx, row in df.iterrows():
                    for c_idx, val in enumerate(row):
                        val_str = str(val)
                        if any(k in val_str.lower() for k in keywords):
                            print(f"    Sheet '{sheet}', Row {r_idx}, Col {df.columns[c_idx]}: {val_str[:150]}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")
