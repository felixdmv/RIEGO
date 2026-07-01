import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
f2024 = os.path.join(folder, "CARRALANCHA ALMENDRO V4 2024 ENERO-JULIO.xlsx")
f2025 = os.path.join(folder, "CARRALANCHA ALMENDRO V4 2025 ENERO-JULIO.xlsx")

for name, filepath in [("2024 ENERO-JULIO", f2024), ("2025 ENERO-JULIO", f2025)]:
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        print(f"\n--- {name} ---")
        # Check rows where 'Riego (cantidad total)' > 0
        df_nz = df[df['Riego (cantidad total)'] > 0]
        if not df_nz.empty:
            print("Muestra con Riego (cantidad total) > 0:")
            print(df_nz[['Fecha', 'Riego (pulsos)', 'Riego (cantidad total)', 'Caudal de riego (cantidad total)', 'Riego diario (cantidad total)']].head(10))
        else:
            print("No hay filas con Riego (cantidad total) > 0")
        
        # Check rows where 'Riego diario (cantidad total)' > 0
        df_rd = df[df['Riego diario (cantidad total)'] > 0]
        if not df_rd.empty:
            print("Muestra con Riego diario (cantidad total) > 0:")
            print(df_rd[['Fecha', 'Riego (pulsos)', 'Riego (cantidad total)', 'Riego diario (cantidad total)']].head(10))
        else:
            print("No hay filas con Riego diario (cantidad total) > 0")
