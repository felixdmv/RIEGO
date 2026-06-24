import pandas as pd
import os

path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela\AEMET_Sardon_historico.csv"
df = pd.read_csv(path, parse_dates=['Fecha'])
print("AEMET columns:", list(df.columns))
print("AEMET shape:", df.shape)
print("AEMET min date:", df['Fecha'].min())
print("AEMET max date:", df['Fecha'].max())
print("\nLast 10 rows in AEMET:")
print(df.tail(10))
