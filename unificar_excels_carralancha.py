import os
import glob
import pandas as pd

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"

clima_files = sorted(glob.glob(os.path.join(folder, "CLIMA CARRALANCHA*.xlsx")))
crop_files = sorted(glob.glob(os.path.join(folder, "CARRALANCHA ALMENDRO V4*.xlsx")))

# 1. Unificar Clima
print("Unificando archivos de clima...")
clima_dfs = []
for f in clima_files:
    print(f"  Leyendo {os.path.basename(f)}...")
    df = pd.read_excel(f, sheet_name=0)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    clima_dfs.append(df)

if clima_dfs:
    df_clima_all = pd.concat(clima_dfs, ignore_index=True)
    df_clima_all = df_clima_all.drop_duplicates(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    out_clima = os.path.join(folder, "CLIMA_CARRALANCHA_sensor_unificado.csv")
    df_clima_all.to_csv(out_clima, index=False)
    print(f"[OK] Guardado clima unificado ({len(df_clima_all)} filas) en {out_clima}")
else:
    print("No se encontraron archivos de clima.")

# 2. Unificar Crop (Almendro V4)
print("\nUnificando archivos de crop (Almendro V4)...")
crop_dfs = []
for f in crop_files:
    print(f"  Leyendo {os.path.basename(f)}...")
    df = pd.read_excel(f, sheet_name=0)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    crop_dfs.append(df)

if crop_dfs:
    df_crop_all = pd.concat(crop_dfs, ignore_index=True)
    df_crop_all = df_crop_all.drop_duplicates(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    out_crop = os.path.join(folder, "CARRALANCHA_ALMENDRO_V4_unificado.csv")
    df_crop_all.to_csv(out_crop, index=False)
    print(f"[OK] Guardado crop unificado ({len(df_crop_all)} filas) en {out_crop}")
else:
    print("No se encontraron archivos de crop.")
