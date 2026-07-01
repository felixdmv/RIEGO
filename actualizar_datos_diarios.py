import os
import pandas as pd
import numpy as np

def update_daily_dataset():
    base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
    
    # 1. Local Clima Sensors Ingestion
    clima_path = os.path.join(base_path, "CLIMA_CARRALANCHA_sensor_unificado.csv")
    if not os.path.exists(clima_path):
        raise FileNotFoundError(f"No se encontró clima unificado en {clima_path}")
        
    print("Procesando clima local...")
    df_sensor = pd.read_csv(clima_path)
    df_sensor['Fecha'] = pd.to_datetime(df_sensor['Fecha'])
    df_clima = df_sensor.set_index('Fecha').resample('D').agg({
        'Temperatura': 'mean',
        'Radiación solar': 'mean',
        'Precipitaciones': 'sum'
    }).reset_index()
    
    # Calculate Reference Evapotranspiration (ET0) using Hargreaves formula
    # Hargreaves: 0.0135 * (T_mean + 17.8) * (Rad * 0.035)
    df_clima['ET0_Calculada'] = 0.0135 * (df_clima['Temperatura'] + 17.8) * (df_clima['Radiación solar'] * 0.035)

    # 2. Local Crop/Soil Sensors & Irrigation Ingestion
    crop_path = os.path.join(base_path, "CARRALANCHA_ALMENDRO_V4_unificado.csv")
    if not os.path.exists(crop_path):
        raise FileNotFoundError(f"No se encontró crop unificado en {crop_path}")
        
    print("Procesando sensores de suelo e irrigación...")
    df_crop_raw = pd.read_csv(crop_path)
    df_crop_raw['Fecha'] = pd.to_datetime(df_crop_raw['Fecha'])
    
    # Define aggregation mapping
    # Humidity -> mean, Potential -> min, Riego -> sum
    df_crop_daily = df_crop_raw.set_index('Fecha').resample('D').agg({
        'Contenido volumétrico (10 cm)': 'mean',
        'Contenido volumétrico (20 cm)': 'mean',
        'Contenido volumétrico (40 cm)': 'mean',
        'Contenido volumétrico (60 cm)': 'mean',
        'Potencial hídrico (almendra)': 'min',
        'Baseline potencial hídrico (almendra)': 'min',
        'Riego (cantidad total)': 'sum'
    }).reset_index()
    
    # Rename columns to match standard schema
    df_crop_daily = df_crop_daily.rename(columns={
        'Contenido volumétrico (10 cm)': 'FP_Humedad_10cm',
        'Contenido volumétrico (20 cm)': 'FP_Humedad_20cm',
        'Contenido volumétrico (40 cm)': 'FP_Humedad_40cm',
        'Contenido volumétrico (60 cm)': 'FP_Humedad_60cm',
        'Potencial hídrico (almendra)': 'FP_Potencial_Min',
        'Baseline potencial hídrico (almendra)': 'FP_Potencial_Base_Min',
        'Riego (cantidad total)': 'Riego_m3_Real'
    })
    
    # Convert irrigation volume (m3) to depth (mm)
    # Area of Sector V4 is 22357.87 m2
    area_v4 = 22357.87
    df_crop_daily['Riego_mm_Real'] = (df_crop_daily['Riego_m3_Real'] * 1000) / area_v4
    
    # Default potential values if they are completely empty (since these columns are mostly empty in V4)
    # We use typical values: Potential Min = -1.5 MPa, Baseline = -0.8 MPa
    if df_crop_daily['FP_Potencial_Min'].isna().all():
        df_crop_daily['FP_Potencial_Min'] = -1.5
    if df_crop_daily['FP_Potencial_Base_Min'].isna().all():
        df_crop_daily['FP_Potencial_Base_Min'] = -0.8

    # 3. AEMET weather forecast (Rueda station 2507Y)
    aemet_path = os.path.join(base_path, "AEMET_Rueda_historico.csv")
    if not os.path.exists(aemet_path):
        raise FileNotFoundError(f"No se encontró AEMET histórico en {aemet_path}")
        
    print("Procesando datos climatológicos AEMET...")
    df_aemet = pd.read_csv(aemet_path, parse_dates=['Fecha'])
    df_aemet = df_aemet.sort_values('Fecha')
    
    # 3-day accumulated rolling rain forecast (standard formula)
    df_aemet['Lluvia_Prevista_3d'] = df_aemet['Lluvia_AEMET'].shift(-1).rolling(3, min_periods=1).sum().shift(-2).fillna(0)

    # 4. Merge base climate, AEMET, Riegos
    print("Fusionando clima, AEMET y riego...")
    df_master = pd.merge(df_clima, df_aemet[['Fecha', 'Lluvia_Prevista_3d', 'Lluvia_AEMET']], on='Fecha', how='outer')
    df_master = pd.merge(df_master, df_crop_daily, on='Fecha', how='outer')
    df_master = df_master.sort_values('Fecha').reset_index(drop=True)

    # 5. Satellite Data Ingest
    print("Procesando e interpolando variables satelitales...")
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
        else:
            print(f"Satélite {col} no encontrado en {path_sat}. Colocando NaN.")
            df_master[col] = np.nan

    # Sort and clip dates to when we have local sensor data (2022-01-01 to 2025-12-31)
    df_master = df_master[(df_master['Fecha'] >= '2022-01-01') & (df_master['Fecha'] <= '2025-12-31')].reset_index(drop=True)

    # Fill NaNs and linear interpolate satellite/sensor data
    df_master['Riego_mm_Real'] = df_master['Riego_mm_Real'].fillna(0)
    df_master['Riego_m3_Real'] = df_master['Riego_m3_Real'].fillna(0)
    df_master['Lluvia_AEMET'] = df_master['Lluvia_AEMET'].fillna(0)
    df_master['Lluvia_Prevista_3d'] = df_master['Lluvia_Prevista_3d'].fillna(0)
    df_master['Precipitaciones'] = df_master['Precipitaciones'].fillna(0)
    
    # Linear interpolation
    cols_to_interpolate = [
        'Temperatura', 'Radiación solar', 'ET0_Calculada',
        'FP_Humedad_10cm', 'FP_Humedad_20cm', 'FP_Humedad_40cm', 'FP_Humedad_60cm',
        'FP_Potencial_Min', 'FP_Potencial_Base_Min',
        'Sat_NDVI', 'Sat_Radar', 'Sat_LST'
    ]
    for col in cols_to_interpolate:
        df_master[col] = df_master[col].interpolate(method='linear', limit_direction='both')
        # Fill any remaining edges
        df_master[col] = df_master[col].bfill().ffill()

    # Kc Dinámico and Crop Evapotranspiration (ETc)
    # Kc = 0.4 + 0.8 * NDVI
    df_master['Kc_Dinamico'] = 0.4 + (df_master['Sat_NDVI'] * 0.8)
    df_master['ETc_Sector4'] = df_master['ET0_Calculada'] * df_master['Kc_Dinamico']

    # Final backup fill for safety
    df_master['FP_Potencial_Min'] = df_master['FP_Potencial_Min'].fillna(-1.5)
    df_master['FP_Potencial_Base_Min'] = df_master['FP_Potencial_Base_Min'].fillna(-0.8)
    
    final_path = os.path.join(base_path, 'Dataset_Final_V4.csv')
    df_master.to_csv(final_path, index=False)
    print(f"\n[OK] Dataset consolidado final V4 guardado en: {final_path}")
    print(f"Filas: {len(df_master)}")
    return df_master

if __name__ == "__main__":
    update_daily_dataset()
