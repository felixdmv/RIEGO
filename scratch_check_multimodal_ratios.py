import pandas as pd

df = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos_multimodales_completos.csv")
nz = df[df['Riego_mm'] > 0]
if not nz.empty:
    print(nz[['Fecha', 'Riego_m3', 'Riego_mm']].head(10))
    ratio = nz['Riego_m3'] / nz['Riego_mm']
    print("\nRatios (Riego_m3 / Riego_mm):")
    print(ratio.describe())
else:
    print("No non-zero irrigation rows found.")
