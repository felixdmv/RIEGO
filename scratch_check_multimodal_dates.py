import pandas as pd

df = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos_multimodales_completos.csv")
print("Rango de Fechas en datos_multimodales_completos.csv:")
print(df['Fecha'].min(), "to", df['Fecha'].max())
print("Filas:", len(df))
print("\nEstadísticas de Riego_m3 y Riego_mm:")
print(df[['Riego_m3', 'Riego_mm']].describe())
