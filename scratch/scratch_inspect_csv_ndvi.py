import pandas as pd
import os

csv_path = r"datos/canduela/satelite/Sentinel2_NDVI_Parcela_Real.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"Sentinel2_NDVI_Parcela_Real.csv:")
    print(f"  Rows count: {len(df)}")
    print(f"  Min date: {df['Fecha'].min()}")
    print(f"  Max date: {df['Fecha'].max()}")
    print(f"  Sample rows:\n{df.head(10)}")
else:
    print("CSV not found.")
