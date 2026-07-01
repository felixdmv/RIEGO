import pandas as pd
import glob
import os

canduela_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela"

def inspect_csv(filename):
    path = os.path.join(canduela_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=5)
        print(f"\nArchivo: {filename}")
        print(f"Columnas: {list(df.columns)}")
        print("Muestra:")
        print(df)
    else:
        print(f"Archivo no existe: {filename}")

inspect_csv("DATOSE_1.csv")
inspect_csv("Dataset_Final_S3.csv")
inspect_csv("RIEGOS_1.csv")
inspect_csv("AEMET_Sardon_historico.csv")
