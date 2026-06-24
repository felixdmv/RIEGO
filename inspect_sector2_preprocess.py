import pandas as pd
import os
import glob
import numpy as np

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\canduela"
area_s2 = 16698.54

print("--- TESTING SECTOR 2 PREPROCESS AND REC FOR 2024-08-10 ---")

# 1. Sensors Data
sensor_files = ['DATOSE_1.csv', 'DATOSE_2.csv', 'DATOSE_3.csv']
list_s = []
for f in sensor_files:
    path = os.path.join(base_path, f)
    if os.path.exists(path):
        df_s = pd.read_csv(path)
        df_s['Fecha'] = pd.to_datetime(df_s['Fecha'], errors='coerce')
        list_s.append(df_s.dropna(subset=['Fecha']))

df_sensor = pd.concat(list_s)
df_clima = df_sensor.set_index('Fecha').resample('D').agg({
    'Temperatura': 'mean',
    'Radiación solar': 'mean',
    'Precipitaciones': 'sum'
}).reset_index()

df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

# 2. AEMET
path_aemet = os.path.join(base_path, 'AEMET_Sardon_historico.csv')
df_aemet = pd.read_csv(path_aemet, parse_dates=['Fecha']).sort_values('Fecha')
df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)

# 3. Irrigation logs (reconstruct Excel for V2)
excel_path = os.path.join(base_path, "CANDUELA - 080221084930-077_ (1).xlsx")
df_xl = pd.read_excel(excel_path)
df_xl['event_id'] = df_xl['Equipo'].notna().cumsum()
comienzo_col = df_xl.columns[4]
consumo_real_col = df_xl.columns[10]
valvulas_col = df_xl.columns[27]

header_cols = [df_xl.columns[i] for i in [0, 2, 4, 5, 6, 9, 10]]
for col in header_cols:
    df_xl[col] = df_xl[col].ffill()

df_v2 = df_xl[df_xl[valvulas_col] == 'V2'].copy()
df_v2['Fecha'] = pd.to_datetime(df_v2[comienzo_col], dayfirst=True, errors='coerce').dt.normalize()
df_v2_events = df_v2.drop_duplicates(subset=['event_id']).copy()
df_s2_grouped = df_v2_events.groupby('Fecha')[consumo_real_col].sum().reset_index()
df_s2_grouped = df_s2_grouped.rename(columns={consumo_real_col: 'Riego_m3_Real'})
df_s2_grouped['Riego_mm_Real'] = (df_s2_grouped['Riego_m3_Real'] * 1000) / area_s2

# 4. Merge
df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_AEMET']], on='Fecha', how='outer')
df_master = pd.merge(df_master, df_s2_grouped[['Fecha', 'Riego_mm_Real']], on='Fecha', how='left').fillna({'Riego_mm_Real': 0})
df_master = df_master.sort_values('Fecha').reset_index(drop=True)

# 5. Satellite Data
archivos_sat = {
    'Sat_NDVI': 'satelite/Sentinel2_NDVI_Parcela_Real.csv',
    'Sat_Radar': 'satelite/Sentinel1_Radar_Parcela_Real.csv',
    'Sat_LST': 'satelite/MODIS_Temp_Parcela_Real.csv'
}
for col, rel_path in archivos_sat.items():
    path_sat = os.path.join(base_path, rel_path)
    if os.path.exists(path_sat):
        df_sat = pd.read_csv(path_sat, parse_dates=['Fecha']).rename(columns={'Valor': col})
        df_master = pd.merge(df_master, df_sat[['Fecha', col]], on='Fecha', how='left')
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
        df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
df_master['ETc_Sector2'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

# 6. Florapulse Data (CANDUELA MINI 2 excels)
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

df_fp_all = pd.concat(dfs_fp, ignore_index=True).sort_values('Fecha')
col_pot = [c for c in df_fp_all.columns if 'potencial' in c.lower() and 'almendra' in c.lower() and 'baseline' not in c.lower()][0]
col_base = [c for c in df_fp_all.columns if 'baseline' in c.lower() and 'almendra' in c.lower()][0]
col_sm10 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '10 cm' in c.lower() and 'sustrato' not in c.lower()][0]
col_sm20 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '20 cm' in c.lower() and 'sustrato' not in c.lower()][0]
col_sm40 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '40 cm' in c.lower() and 'sustrato' not in c.lower()][0]
col_sm60 = [c for c in df_fp_all.columns if 'volum' in c.lower() and '60 cm' in c.lower() and 'sustrato' not in c.lower()][0]

agg_dict = {
    col_pot: 'min', col_base: 'min',
    col_sm10: 'mean', col_sm20: 'mean', col_sm40: 'mean', col_sm60: 'mean'
}
df_fp_daily = df_fp_all.set_index('Fecha').resample('D').agg(agg_dict).reset_index()
rename_dict = {
    col_pot: 'FP_Potencial_Min', col_base: 'FP_Potencial_Base_Min',
    col_sm10: 'FP_Humedad_10cm', col_sm20: 'FP_Humedad_20cm',
    col_sm40: 'FP_Humedad_40cm', col_sm60: 'FP_Humedad_60cm'
}
df_fp_daily = df_fp_daily.rename(columns=rename_dict)
df_master = pd.merge(df_master, df_fp_daily, on='Fecha', how='left')

fp_cols = ['FP_Potencial_Min', 'FP_Potencial_Base_Min', 'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm']
for col in fp_cols:
    df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')

df_master['Temperatura'] = df_master['Temperatura'].interpolate(method='linear', limit_direction='both')
df_master['Radiación solar'] = df_master['Radiación solar'].interpolate(method='linear', limit_direction='both')
df_master['Precipitaciones'] = df_master['Precipitaciones'].fillna(0)
df_master['ET0_Calculada'] = df_master['ET0_Calculada'].interpolate(method='linear', limit_direction='both')
df_master['ETc_Sector2'] = df_master['ETc_Sector2'].interpolate(method='linear', limit_direction='both')

# Drop any remaining NaNs in essential columns
df_master = df_master.dropna(subset=['Sat_NDVI', 'Sat_Radar', 'Sat_LST', 'FP_Potencial_Min']).reset_index(drop=True)

print("\nPreprocessed dataset shape:", df_master.shape)

# Let's find 2024-08-10 row
target_date = pd.to_datetime('2024-08-10')
matching_rows = df_master[df_master['Fecha'] == target_date]
if len(matching_rows) > 0:
    row = matching_rows.iloc[0]
    print(f"\n--- DATA FOR {target_date.strftime('%Y-%m-%d')} ---")
    print(f"  NDVI: {row['Sat_NDVI']:.3f}")
    print(f"  Kc: {row['Kc_Dinamico']:.3f}")
    print(f"  ET0: {row['ET0_Calculada']:.3f} mm/day")
    print(f"  ETc: {row['ETc_Sector2']:.3f} mm/day")
    print(f"  Lluvia prevista 3d: {row['Lluvia_Prevista_3d']:.1f} mm")
    print(f"  Potencial hídrico: {row['FP_Potencial_Min']:.2f} bar")
    print(f"  Humedad suelo 10cm: {row['FP_Humedad_10cm']:.2f}%")
    print(f"  Humedad suelo 40cm: {row['FP_Humedad_40cm']:.2f}%")
    
    # Recommendation calculation
    etc = row['ETc_Sector2']
    lluvia_p = row['Lluvia_Prevista_3d']
    pot = row['FP_Potencial_Min']
    
    dosis_ideal_base = etc * 3 - lluvia_p
    
    # RDI Strategy
    if pot >= -8.0:
        coef = 0.70
        strategy = "Suelo húmedo y potencial óptimo. Se prescribe riego de mantenimiento para mantener el nivel actual de vigor."
    elif -12.0 <= pot < -8.0:
        coef = 0.60
        strategy = "Estrés leve favorable para el cultivo en esta etapa. Mantener estrategia de RDI con dosis controlada."
    elif -16.0 <= pot < -12.0:
        coef = 0.85
        strategy = "Se detecta estrés moderado en FloraPulse. Incrementar dosis de riego para recuperar el potencial hídrico foliar."
        dosis_ideal_base += 3.0
    else:
        coef = 1.00
        strategy = "ALERTA CRÍTICA: Potencial hídrico en zona de peligro. Aplicar riego de choque de forma inmediata para restaurar turgencia celular."
        dosis_ideal_base += 8.0
        
    dosis_rec = max(0, dosis_ideal_base * coef)
    volumen_m3 = (dosis_rec * area_s2) / 1000.0
    
    # Caudal nominal is 1.09 mm/h for Sector 2
    precipitation_rate = 1.09
    irr_hours = dosis_rec / precipitation_rate
    
    print("\n--- RECOMMENDATION FOR SECTOR 2 ---")
    print(f"  Dosis Recomendada: {dosis_rec:.2f} mm")
    print(f"  Volumen Total: {volumen_m3:.2f} m3")
    print(f"  Tiempo de Riego: {irr_hours:.2f} horas (at 1.09 mm/h)")
    print(f"  Estrategia: {strategy}")
else:
    print(f"\n[ERROR] Target date {target_date.strftime('%Y-%m-%d')} not found in dataset!")
