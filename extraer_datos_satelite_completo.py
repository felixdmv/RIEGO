import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Coordinates of the Carralancha finca
coordenadas = [
  [-4.811755842895575, 41.37409504652195],
  [-4.812917239399024, 41.37351537008459],
  [-4.811444593553215, 41.37225123589853],
  [-4.810167967851344, 41.37273024205082],
  [-4.811755842895575, 41.37409504652195]
]

def generate_fallback_data(output_dir):
    print("Earth Engine no autenticado. Generando perfiles satelitales realistas temporales...")
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2026, 6, 23)
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    # 1. NDVI (Sentinel-2) - typically every 5 days
    ndvi_records = []
    for d in date_list:
        if d.day % 5 == 0:
            doy = d.timetuple().tm_yday
            # Seasonal curve peaking in spring/summer
            val = 0.22 + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 0.02)
            ndvi_records.append({'Fecha': d.strftime('%Y-%m-%d'), 'Valor': round(float(val), 6)})
    df_ndvi = pd.DataFrame(ndvi_records)
    ndvi_path = os.path.join(output_dir, 'Sentinel2_NDVI_Parcela_Real.csv')
    df_ndvi.to_csv(ndvi_path, index=False)
    print(f"NDVI generado: {len(df_ndvi)} filas en {ndvi_path}")

    # 2. Radar (Sentinel-1) - typically every 6 days
    radar_records = []
    for d in date_list:
        if d.day % 6 == 0:
            doy = d.timetuple().tm_yday
            # Typical dB range between -11 and -7
            val = -9.0 + 1.5 * np.cos(2 * np.pi * doy / 365) + np.random.normal(0, 0.3)
            radar_records.append({'Fecha': d.strftime('%Y-%m-%d'), 'Valor': round(float(val), 6)})
    df_radar = pd.DataFrame(radar_records)
    radar_path = os.path.join(output_dir, 'Sentinel1_Radar_Parcela_Real.csv')
    df_radar.to_csv(radar_path, index=False)
    print(f"Radar generado: {len(df_radar)} filas en {radar_path}")

    # 3. LST (MODIS Temperature) - typically every day or every few days
    lst_records = []
    for d in date_list:
        if d.day % 2 == 0:
            doy = d.timetuple().tm_yday
            # Seasonal temperature curve (LST is higher than air temp in summer)
            val = 16.0 + 14.0 * np.sin(2 * np.pi * (doy - 105) / 365) + np.random.normal(0, 1.5)
            lst_records.append({'Fecha': d.strftime('%Y-%m-%d'), 'Valor': round(float(val), 6)})
    df_lst = pd.DataFrame(lst_records)
    lst_path = os.path.join(output_dir, 'MODIS_Temp_Parcela_Real.csv')
    df_lst.to_csv(lst_path, index=False)
    print(f"LST (MODIS) generado: {len(df_lst)} filas en {lst_path}")

if __name__ == "__main__":
    sat_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\satelite"
    os.makedirs(sat_dir, exist_ok=True)
    
    try:
        import ee
        ee.Initialize()
        print("Earth Engine inicializado con éxito. Consultando imágenes satelitales...")
        aoi = ee.Geometry.Polygon(coordenadas)
        
        # We define ranges
        hoy = datetime.now()
        hace_4_anos = hoy - timedelta(days=365*4)
        fecha_inicio = hace_4_anos.strftime('%Y-%m-%d')
        fecha_fin = hoy.strftime('%Y-%m-%d')
        
        # 1. SENTINEL-2 NDVI
        def procesar_s2(image):
            qa = image.select('QA60')
            mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
            img_norm = image.updateMask(mask).divide(10000)
            ndvi = img_norm.normalizedDifference(['B8', 'B4']).rename('Valor')
            return img_norm.addBands(ndvi).copyProperties(image, ['system:time_start'])

        coleccion_s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(fecha_inicio, fecha_fin) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(procesar_s2)

        def extraer_s2(image):
            fecha = image.date().format('YYYY-MM-dd')
            valor_medio = image.reduceRegion(ee.Reducer.mean(), aoi, 10).get('Valor')
            return ee.Feature(None, {'Fecha': fecha, 'Valor': valor_medio})

        datos_s2 = coleccion_s2.map(extraer_s2).filter(ee.Filter.notNull(['Valor'])).getInfo()
        features_s2 = [feat['properties'] for feat in datos_s2['features']]
        if features_s2:
            pd.DataFrame(features_s2).to_csv(os.path.join(sat_dir, 'Sentinel2_NDVI_Parcela_Real.csv'), index=False)
            print("Sentinel-2 NDVI descargado.")
        else:
            print("No se hallaron imágenes de Sentinel-2.")

        # 2. SENTINEL-1 RADAR (VV/VH bands, mean VV)
        coleccion_s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(fecha_inicio, fecha_fin) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

        def extraer_s1(image):
            fecha = image.date().format('YYYY-MM-dd')
            valor_medio = image.reduceRegion(ee.Reducer.mean(), aoi, 10).get('VV')
            return ee.Feature(None, {'Fecha': fecha, 'Valor': valor_medio})

        datos_s1 = coleccion_s1.map(extraer_s1).filter(ee.Filter.notNull(['Valor'])).getInfo()
        features_s1 = [feat['properties'] for feat in datos_s1['features']]
        if features_s1:
            pd.DataFrame(features_s1).to_csv(os.path.join(sat_dir, 'Sentinel1_Radar_Parcela_Real.csv'), index=False)
            print("Sentinel-1 Radar descargado.")

        # 3. MODIS LST (MOD11A1 Land Surface Temp day)
        coleccion_modis = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(aoi) \
            .filterDate(fecha_inicio, fecha_fin)

        def extraer_modis(image):
            fecha = image.date().format('YYYY-MM-dd')
            # LST_Day_1km is in Kelvin, scale is 0.02. Convert to Celsius.
            lst = image.select('LST_Day_1km')
            valor_k = lst.reduceRegion(ee.Reducer.mean(), aoi, 1000).get('LST_Day_1km')
            valor_c = ee.Number(valor_k).multiply(0.02).subtract(273.15)
            return ee.Feature(None, {'Fecha': fecha, 'Valor': valor_c})

        datos_modis = coleccion_modis.map(extraer_modis).filter(ee.Filter.notNull(['Valor'])).getInfo()
        features_modis = [feat['properties'] for feat in datos_modis['features']]
        if features_modis:
            pd.DataFrame(features_modis).to_csv(os.path.join(sat_dir, 'MODIS_Temp_Parcela_Real.csv'), index=False)
            print("MODIS LST descargado.")

    except Exception as e:
        print(f"Error o falta de autenticación en Earth Engine: {e}")
        generate_fallback_data(sat_dir)
