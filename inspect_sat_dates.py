import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela"
files = {
    'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
    'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
    'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
}
for name, rel in files.items():
    path = os.path.join(base_path, rel)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['Fecha'])
        print(f"\n{name} columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Min Date: {df['Fecha'].min()}")
        print(f"  Max Date: {df['Fecha'].max()}")
        print(f"  Nulls: {df['Valor'].isna().sum()}")
        print("  Latest rows:")
        print(df.tail(3))
    else:
        print(f"\n{name} path {path} does not exist!")
