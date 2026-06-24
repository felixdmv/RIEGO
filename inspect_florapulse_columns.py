import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
fp_file = os.path.join(base_path, "datos", "canduela", "florapulse", "CANDUELA MINI 2 ALMENDRO 2024 ENERO-JULIO.xlsx")

print("File path:", fp_file)
print("File exists:", os.path.exists(fp_file))

xl = pd.ExcelFile(fp_file)
print("Sheets:", xl.sheet_names)
df = xl.parse(xl.sheet_names[0])
print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
print("First 3 rows:")
print(df.head(3))
