import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)

print("--- SEARCH FOR METADATA IN EXCEL VALUES ---")
keywords = ["caudal", "superficie", "area", "flow", "m2", "ha", "litro", "m3/h", "l/h", "diseño", "diseno", "poligono", "polygon", "sector 2", "sector2", "v2"]
found = False

for r_idx, row in df.iterrows():
    for c_idx, val in enumerate(row):
        val_str = str(val)
        if any(k in val_str.lower() for k in keywords):
            print(f"Row {r_idx}, Col {df.columns[c_idx]} (Index {c_idx}): {val_str}")
            found = True

if not found:
    print("No keywords found in cell values.")
