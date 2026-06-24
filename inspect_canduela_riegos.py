import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# 1. RIEGOS_1.csv
riegos_path = os.path.join(base_path, "datos", "canduela", "RIEGOS_1.csv")
print("--- RIEGOS_1.csv ---")
if os.path.exists(riegos_path):
    df_r = pd.read_csv(riegos_path)
    print("Columns:", list(df_r.columns))
    print("Unique Sectores de Riego:", df_r['Sectores de Riego'].dropna().unique())
    print("Head:")
    print(df_r.head(10))
else:
    print("RIEGOS_1.csv does not exist")

# 2. Search for "Sector 2" or coordinates in CANDUELA - 080221084930-077_ (1).xlsx
print("\n--- Excel details ---")
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")
df_xl = pd.read_excel(excel_path)
print("Unique 'Válvulas':", df_xl['Válvulas'].dropna().unique())
print("Unique 'Equipo':", df_xl['Equipo'].dropna().unique())
print("Unique 'Finca':", df_xl['Finca'].dropna().unique())
print("Unique 'Programa':", df_xl['Programa'].dropna().unique())
# Look for rows where Válvulas is 'V2' or similar
df_v2 = df_xl[df_xl['Válvulas'].astype(str).str.contains('V2|2', case=False, na=False)]
print("V2 rows count:", len(df_v2))
if len(df_v2) > 0:
    print("V2 sample rows:")
    print(df_v2[['Comienzo', 'Finalización', 'Duración', 'Consumo programado (m3)', 'Consumo real (m3)', 'Válvulas']].head(5))
