import glob
import os
import pandas as pd

base_path = "datos/canduela"

# 1. Cesens
print("Cesens excels:")
for f in sorted(glob.glob(os.path.join(base_path, "Datos estaciones Cesens *.xlsx"))):
    xl = pd.ExcelFile(f)
    df_s = xl.parse(xl.sheet_names[0])
    df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
    df_s = df_s.dropna(subset=['Fecha'])
    print(f"  {os.path.basename(f)}: max date = {df_s['Fecha'].max()}")

# 2. FloraPulse
print("\nFloraPulse excels:")
for f in sorted(glob.glob(os.path.join(base_path, "florapulse", "*.xlsx"))):
    xl = pd.ExcelFile(f)
    df_f = xl.parse(xl.sheet_names[0])
    fecha_col = [c for c in df_f.columns if 'fecha' in str(c).lower()][0]
    df_f['Fecha'] = pd.to_datetime(df_f[fecha_col], errors='coerce')
    df_f = df_f.dropna(subset=['Fecha'])
    print(f"  {os.path.basename(f)}: max date = {df_f['Fecha'].max()}")

# 3. AEMET
aemet_path = os.path.join(base_path, "AEMET_Sardon_historico.csv")
if os.path.exists(aemet_path):
    df_aemet = pd.read_csv(aemet_path)
    df_aemet['Fecha'] = pd.to_datetime(df_aemet['Fecha'])
    print(f"\nAEMET Sardon CSV max date = {df_aemet['Fecha'].max()}")
else:
    print("\nAEMET Sardon CSV not found.")
