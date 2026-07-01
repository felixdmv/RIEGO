import pandas as pd
import os

csv_path = r"datos/canduela/Dataset_Final_S2.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"Dataset_Final_S2.csv:")
    print(f"  Shape: {df.shape}")
    print(f"  Min date: {df['Fecha'].min()}")
    print(f"  Max date: {df['Fecha'].max()}")
    print(f"  Latest rows:\n{df.tail(5)[['Fecha', 'Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 'Riego_mm_Real', 'Lluvia_AEMET']]}")
else:
    print(f"Dataset_Final_S2.csv not found at {csv_path}")

csv_path_s3 = r"datos/canduela/Dataset_Final_S3.csv"
if os.path.exists(csv_path_s3):
    df_s3 = pd.read_csv(csv_path_s3)
    df_s3['Fecha'] = pd.to_datetime(df_s3['Fecha'])
    print(f"\nDataset_Final_S3.csv:")
    print(f"  Shape: {df_s3.shape}")
    print(f"  Min date: {df_s3['Fecha'].min()}")
    print(f"  Max date: {df_s3['Fecha'].max()}")
else:
    print(f"Dataset_Final_S3.csv not found at {csv_path_s3}")
