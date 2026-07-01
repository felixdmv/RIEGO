import pandas as pd

# Canduela Final Dataset
df_cand = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela\Dataset_Final_S3.csv")
print("CANDUELA Riego_mm_Real stats:")
non_zero_cand = df_cand[df_cand['Riego_mm_Real'] > 0]['Riego_mm_Real']
print(f"Count: {len(non_zero_cand)}")
print(f"Min: {non_zero_cand.min()}")
print(f"Max: {non_zero_cand.max()}")
print(f"Mean: {non_zero_cand.mean()}")

# Carralancha Crop Dataset
df_carr = pd.read_csv(r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\CARRALANCHA_ALMENDRO_V4_unificado.csv")
print("\nCARRALANCHA Riego diario (cantidad total) stats:")
# Riego diario (cantidad total) has daily sums, typically populated once a day (at 00:00:00)
df_carr_daily = df_carr[df_carr['Riego diario (cantidad total)'] > 0]
print(f"Count: {len(df_carr_daily)}")
print(df_carr_daily[['Fecha', 'Riego diario (cantidad total)']].head(20))
print(f"Min: {df_carr_daily['Riego diario (cantidad total)'].min()}")
print(f"Max: {df_carr_daily['Riego diario (cantidad total)'].max()}")
print(f"Mean: {df_carr_daily['Riego diario (cantidad total)'].mean()}")
