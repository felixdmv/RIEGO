import os
import re
import pandas as pd
import numpy as np
import glob
from datetime import datetime

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')

# 1. Sensors Data
print("Loading Sensors...")
archivos_sensores = ['DATOSE_1.csv', 'DATOSE_2.csv', 'DATOSE_3.csv']
list_s = []
for f in archivos_sensores:
    df_s = pd.read_csv(os.path.join(datos_path, f))
    df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
    list_s.append(df_s.dropna(subset=['Fecha']))
df_sensor = pd.concat(list_s)
df_clima = df_sensor.set_index('Fecha').resample('D').agg({
    'Temperatura': 'mean', 
    'Radiación solar': 'mean', 
    'Precipitaciones': 'sum'
}).reset_index()

# Hargreaves Reference Evapotranspiration
df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

# 2. AEMET Weather
print("Loading AEMET weather...")
df_aemet = pd.read_csv(os.path.join(datos_path, 'AEMET_Sardon_historico.csv'), parse_dates=['Fecha'])
df_aemet = df_aemet.sort_values('Fecha')
df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)

# 3. Riegos (Irrigation) Sector 3
print("Loading Irrigation logs...")
df_r = pd.read_csv(os.path.join(datos_path, 'RIEGOS_1.csv'))
df_r['Sectores de Riego'] = pd.to_numeric(df_r['Sectores de Riego'], errors='coerce')
df_r['Agua m3'] = df_r['Agua m³'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_r['Agua m3'] = pd.to_numeric(df_r['Agua m3'], errors='coerce')
df_r['Fecha'] = pd.to_datetime(df_r['Inicio'].astype(str).str[:10], dayfirst=True, errors='coerce')
area_s3 = 40650
df_s3 = df_r[df_r['Sectores de Riego'] == 3].groupby('Fecha')['Agua m3'].sum().reset_index()
df_s3['Riego_mm_Real'] = (df_s3['Agua m3'] * 1000) / area_s3

# 4. Merge Sensor, AEMET, Riegos
df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_AEMET']], on='Fecha', how='outer')
df_master = pd.merge(df_master, df_s3[['Fecha', 'Riego_mm_Real']], on='Fecha', how='left').fillna({'Riego_mm_Real': 0})
df_master = df_master.sort_values('Fecha').reset_index(drop=True)

# 5. Satellite Data
print("Loading Satellites...")
archivos_sat = {
    'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
    'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
    'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
}
for col, path in archivos_sat.items():
    df_sat = pd.read_csv(os.path.join(datos_path, path), parse_dates=['Fecha']).rename(columns={'Valor': col})
    df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
    df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

# Kc Dinámico and Crop Evapotranspiration
df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
df_master['ETc_Sector3'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

# 6. Florapulse Data Loading and Aggregation
print("Loading Florapulse Excel sheets...")
excel_files = glob.glob(os.path.join(datos_path, 'florapulse', "*.xlsx"))
dfs_fp = []
for f in excel_files:
    try:
        xl = pd.ExcelFile(f)
        df_f = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df_f.columns if 'fecha' in str(c).lower()][0]
        df_f = df_f.rename(columns={fecha_col: 'Fecha'})
        df_f['Fecha'] = pd.to_datetime(df_f['Fecha'], errors='coerce')
        df_f = df_f.dropna(subset=['Fecha'])
        
        # Keep numeric cols
        cols_to_convert = [c for c in df_f.columns if c != 'Fecha']
        for c in cols_to_convert:
            df_f[c] = pd.to_numeric(df_f[c], errors='coerce')
        
        dfs_fp.append(df_f)
        print(f"  Loaded {os.path.basename(f)}: {df_f.shape}")
    except Exception as e:
        print(f"  Error loading {os.path.basename(f)}: {e}")

if dfs_fp:
    df_fp_all = pd.concat(dfs_fp, ignore_index=True)
    df_fp_all = df_fp_all.sort_values('Fecha')
    
    # We will aggregate 'Potencial hídrico (almendra)' and 'Baseline potencial hídrico (almendra)' by taking the MINIMUM (representing peak stress)
    # And other variables (soil moisture, temperature) by taking the MEAN.
    
    # We will find the exact column names by matching keywords.
    col_pot = [c for c in df_fp_all.columns if 'potencial' in c.lower() and 'almendra' in c.lower() and 'baseline' not in c.lower()][0]
    col_base = [c for c in df_fp_all.columns if 'baseline' in c.lower() and 'almendra' in c.lower()][0]
    col_sm10 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '10 cm' in c.lower() and 'sustrato' not in c.lower()][0]
    col_sm20 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '20 cm' in c.lower() and 'sustrato' not in c.lower()][0]
    col_sm40 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '40 cm' in c.lower() and 'sustrato' not in c.lower()][0]
    col_sm60 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '60 cm' in c.lower() and 'sustrato' not in c.lower()][0]
    
    # Let's define the aggregation dictionary
    agg_dict = {
        col_pot: 'min',
        col_base: 'min',
        col_sm10: 'mean',
        col_sm20: 'mean',
        col_sm40: 'mean',
        col_sm60: 'mean'
    }
    
    # Aggregate to daily
    df_fp_daily = df_fp_all.set_index('Fecha').resample('D').agg(agg_dict).reset_index()
    
    # Rename columns to standard names
    rename_dict = {
        col_pot: 'FP_Potencial_Min',
        col_base: 'FP_Potencial_Base_Min',
        col_sm10: 'FP_Humedad_10cm',
        col_sm20: 'FP_Humedad_20cm',
        col_sm40: 'FP_Humedad_40cm',
        col_sm60: 'FP_Humedad_60cm'
    }
    df_fp_daily = df_fp_daily.rename(columns=rename_dict)
    
    print("\nFlorapulse daily aggregated shape:", df_fp_daily.shape)
    print("Columns after rename:", df_fp_daily.columns.tolist())
    
    # 7. Merge Florapulse with Master
    print("\nMerging Florapulse into master...")
    df_master = pd.merge(df_master, df_fp_daily, on='Fecha', how='left')
    
    # Interpolate Florapulse columns to fill small gaps
    fp_cols = ['FP_Potencial_Min', 'FP_Potencial_Base_Min', 'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm']
    for col in fp_cols:
        df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

print("\nFinal Master Dataset shape:", df_master.shape)
print("Final Columns:", df_master.columns.tolist())
print("Missing values per column in final dataset:")
print(df_master.isna().sum())
print("\nFirst 5 rows of merged dataset:")
print(df_master.head(5))
print("\nChecking dates where both NDVI and Florapulse are present:")
overlap_df = df_master.dropna(subset=['Sat_NDVI', 'FP_Potencial_Min'])
print("Overlap rows:", len(overlap_df))
print(overlap_df[['Fecha', 'Sat_NDVI', 'FP_Potencial_Min', 'Riego_mm_Real']].head(5))
