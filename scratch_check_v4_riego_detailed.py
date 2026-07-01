import pandas as pd

df = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\CARRALANCHA_ALMENDRO_V4_unificado.csv")
df['Fecha'] = pd.to_datetime(df['Fecha'])

cols = ['Fecha', 'Riego (pulsos)', 'Riego (cantidad total)', 'Caudal de riego (cantidad total)', 'Riego diario (cantidad total)', 'Riego (por superficie)', 'Riego diario (por superficie)']

# Filtrar filas donde al menos un riego sea > 0
df_nz = df[(df['Riego (cantidad total)'] > 0) | (df['Riego diario (cantidad total)'] > 0)]

print(f"Número de filas con riego > 0: {len(df_nz)}")
if not df_nz.empty:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_nz[cols].head(30))
