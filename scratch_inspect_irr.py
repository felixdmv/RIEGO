import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
crop_file = os.path.join(folder, "CARRALANCHA ALMENDRO V4 2022 AGOSTO-DICIEMBRE.xlsx")

if os.path.exists(crop_file):
    df = pd.read_excel(crop_file, sheet_name=0)
    print("Columnas de riego y potencial:")
    irr_cols = [c for c in df.columns if 'riego' in c.lower() or 'potencial' in c.lower() or 'humedad' in c.lower() or 'volum' in c.lower()]
    print(irr_cols)
    print("\nEstadísticas y no nulos:")
    for col in irr_cols:
        non_null = df[col].dropna()
        print(f"{col}: {len(non_null)} no nulos, min={non_null.min() if len(non_null)>0 else 'N/A'}, max={non_null.max() if len(non_null)>0 else 'N/A'}, mean={non_null.mean() if len(non_null)>0 else 'N/A'}")
    
    print("\nMuestra de las primeras filas con riego > 0:")
    riego_col = 'Riego (cantidad total)'
    if riego_col in df.columns:
        df_r = df[df[riego_col] > 0]
        if not df_r.empty:
            print(df_r[['Fecha', 'Riego (pulsos)', 'Riego (cantidad total)', 'Riego diario (cantidad total)', 'Riego (por superficie)', 'Riego diario (por superficie)']].head(10))
        else:
            print("No hay filas con Riego (cantidad total) > 0")
else:
    print(f"Archivo no existe: {crop_file}")
