import pandas as pd

df = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\CARRALANCHA_ALMENDRO_V4_unificado.csv")
df['Fecha'] = pd.to_datetime(df['Fecha'])

cols = [c for c in df.columns if 'riego' in c.lower()]
print("Columnas de riego:")
print(cols)

# Filtrar filas donde al menos un riego sea > 0
df_nz = df[(df['Riego (cantidad total)'] > 0) | (df['Riego diario (cantidad total)'] > 0) | (df['Riego (por superficie)'] > 0)]

print(f"\nNúmero de filas con riego > 0: {len(df_nz)}")
if not df_nz.empty:
    print("\nMuestra de filas con riego > 0:")
    # Mostrar las primeras 20 filas ordenadas
    print(df_nz[['Fecha', 'Riego (pulsos)', 'Riego (cantidad total)', 'Caudal de riego (cantidad total)', 'Riego diario (cantidad total)', 'Riego (por superficie)', 'Riego diario (por superficie)']].dropna(how='all').head(20))
