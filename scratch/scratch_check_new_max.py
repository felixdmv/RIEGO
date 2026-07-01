import pandas as pd
import os

csv_path = r"datos/canduela/Dataset_Final_S2.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"New Dataset_Final_S2.csv:")
    print(f"  Shape: {df.shape}")
    print(f"  Min date: {df['Fecha'].min()}")
    print(f"  Max date: {df['Fecha'].max()}")
    print(f"  Latest rows:\n{df.tail(10)[['Fecha', 'Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 'Riego_mm_Real', 'Lluvia_AEMET']]}")
else:
    print("CSV not found.")
