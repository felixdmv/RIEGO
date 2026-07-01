import pandas as pd
import os

path = "datos/canduela/AEMET_Sardon_historico.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"Shape: {df.shape}")
    print("Latest 5 rows:")
    print(df.tail(5))
else:
    print("File not found.")
