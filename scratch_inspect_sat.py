import pandas as pd
import os

sat_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela\satelite"
for f in os.listdir(sat_dir):
    p = os.path.join(sat_dir, f)
    df = pd.read_csv(p)
    print(f"\nArchivo: {f}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Rango de Fechas: {df['Fecha'].min()} to {df['Fecha'].max()} (filas: {len(df)})")
    print("Muestra:")
    print(df.head(3))
