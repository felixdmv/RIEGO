import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
crop_files = glob.glob(os.path.join(folder, "CARRALANCHA ALMENDRO V4*.xlsx"))

for f in sorted(crop_files):
    name = os.path.basename(f)
    try:
        df = pd.read_excel(f, sheet_name=0)
        print(f"\nArchivo: {name}")
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0 and ('riego' in col.lower() or 'potencial' in col.lower() or 'humedad' in col.lower() or 'volum' in col.lower()):
                print(f"  {col}: {len(non_null)} no nulos, min={non_null.min()}, max={non_null.max()}")
    except Exception as e:
        print(f"Error en {name}: {e}")
