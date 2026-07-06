import pandas as pd
import glob
import os

def check_xlsx(p):
    try:
        xl = pd.ExcelFile(p)
        df = xl.parse(xl.sheet_names[0])
        fcol = [c for c in df.columns if 'fecha' in str(c).lower() or 'comienzo' in str(c).lower() or 'inicio' in str(c).lower()]
        if fcol:
            df['Parsed'] = pd.to_datetime(df[fcol[0]], errors='coerce')
            print(f"{os.path.basename(p)}: Min={df['Parsed'].min()}, Max={df['Parsed'].max()}, rows={len(df)}")
        else:
            print(f"{os.path.basename(p)}: No date col found. Cols: {list(df.columns)[:5]}")
    except Exception as e:
        print(f"{os.path.basename(p)}: Error {e}")

print("--- EXCEL FILES ---")
check_xlsx('datos/canduela/CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx')
check_xlsx('datos/canduela/Datos estaciones Cesens 2026.xlsx')

print("\n--- SATELLITE FILES ---")
for sat in glob.glob('datos/canduela/satelite/*.csv'):
    try:
        df = pd.read_csv(sat)
        print(f"{os.path.basename(sat)}: Min={df['Fecha'].min()}, Max={df['Fecha'].max()}, rows={len(df)}")
    except Exception as e:
         print(f"{os.path.basename(sat)}: Error {e}")

print("\n--- AEMET FILE ---")
try:
    df_aemet = pd.read_csv('datos/canduela/AEMET_Sardon_historico.csv')
    print(f"AEMET_Sardon_historico.csv: Min={df_aemet['Fecha'].min()}, Max={df_aemet['Fecha'].max()}, rows={len(df_aemet)}")
except Exception as e:
    print(f"AEMET: Error {e}")
