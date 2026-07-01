import os
import glob
import pandas as pd
import numpy as np

base_path = "datos/canduela"
sector = 2
area_s3 = 16698.54

# 1. Sensors Data (Cesens Excel files)
excel_files = glob.glob(os.path.join(base_path, "Datos estaciones Cesens *.xlsx"))
list_s = []
for f in sorted(excel_files):
    xl = pd.ExcelFile(f)
    sheet = xl.sheet_names[0]
    df_s = xl.parse(sheet)
    df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
    df_s = df_s.dropna(subset=['Fecha'])
    df_s = df_s[['Fecha', 'Temperatura', 'Radiación solar', 'Precipitaciones']]
    list_s.append(df_s)
df_sensor = pd.concat(list_s, ignore_index=True)
df_clima = df_sensor.set_index('Fecha').resample('D').agg({
    'Temperatura': 'mean',
    'Radiación solar': 'mean',
    'Precipitaciones': 'sum'
}).reset_index()
df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

# 2. AEMET weather forecast
path_aemet = os.path.join(base_path, "AEMET_Sardon_historico.csv")
df_aemet = pd.read_csv(path_aemet, parse_dates=['Fecha']).sort_values('Fecha')
df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)
df_aemet['Lluvia_Prevista_5d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(5, min_periods=1).sum().shift(-4).fillna(0)

# Merge climate, AEMET
df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_Prevista_5d', 'Lluvia_AEMET']], on='Fecha', how='outer')

# 3. Satellite Data Ingest
archivos_sat = {
    'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
    'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
    'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
}
for col, rel_path in archivos_sat.items():
    path_sat = os.path.join(base_path, rel_path)
    if os.path.exists(path_sat):
        df_sat = pd.read_csv(path_sat, parse_dates=['Fecha']).rename(columns={'Valor': col})
        df_sat = df_sat.groupby('Fecha')[col].mean().reset_index()
        df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
        df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
df_master['ETc_Sector2'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

# 4. FloraPulse Data Ingestion
excel_files = glob.glob(os.path.join(base_path, 'florapulse', "*.xlsx"))
dfs_fp = []
for f in excel_files:
    xl = pd.ExcelFile(f)
    df_f = xl.parse(xl.sheet_names[0])
    fecha_col = [c for c in df_f.columns if 'fecha' in str(c).lower()][0]
    df_f = df_f.rename(columns={fecha_col: 'Fecha'})
    df_f['Fecha'] = pd.to_datetime(df_f['Fecha'], errors='coerce')
    df_f = df_f.dropna(subset=['Fecha'])
    cols_to_convert = [c for c in df_f.columns if c != 'Fecha']
    for c in cols_to_convert:
        df_f[c] = pd.to_numeric(df_f[c], errors='coerce')
    dfs_fp.append(df_f)

if dfs_fp:
    df_fp_all = pd.concat(dfs_fp, ignore_index=True).sort_values('Fecha')
    col_pot = [c for c in df_fp_all.columns if 'potencial' in c.lower() and 'almendra' in c.lower() and 'baseline' not in c.lower()][0]
    col_base = [c for c in df_fp_all.columns if 'baseline' in c.lower() and 'almendra' in c.lower()][0]
    col_sm10 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '10 cm' in c.lower()][0]
    col_sm20 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '20 cm' in c.lower()][0]
    col_sm40 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '40 cm' in c.lower()][0]
    col_sm60 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '60 cm' in c.lower()][0]
    agg_dict = {col_pot: 'min', col_base: 'min', col_sm10: 'mean', col_sm20: 'mean', col_sm40: 'mean', col_sm60: 'mean'}
    df_fp_daily = df_fp_all.set_index('Fecha').resample('D').agg(agg_dict).reset_index()
    df_fp_daily = df_fp_daily.rename(columns={
        col_pot: 'FP_Potencial_Min', col_base: 'FP_Potencial_Base_Min',
        col_sm10: 'FP_Humedad_10cm', col_sm20: 'FP_Humedad_20cm',
        col_sm40: 'FP_Humedad_40cm', col_sm60: 'FP_Humedad_60cm'
    })
    df_master = pd.merge(df_master, df_fp_daily, on='Fecha', how='left')
    fp_cols = ['FP_Potencial_Min', 'FP_Potencial_Base_Min', 'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm']
    for col in fp_cols:
        df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

print("BEFORE dropna (subset of interest):")
subset = df_master[(df_master['Fecha'] >= '2026-06-22') & (df_master['Fecha'] <= '2026-06-26')]
print(subset[['Fecha', 'Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 'Temperatura']])

print("\nNaN counts for each of these columns on June 25:")
row_25 = df_master[df_master['Fecha'] == '2026-06-25']
if not row_25.empty:
    print(row_25[['Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min', 'Temperatura']].isna().sum())
else:
    print("Row for June 25 is empty in df_master!")
