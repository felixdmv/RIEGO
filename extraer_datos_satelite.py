import os
import ee
import pandas as pd
from datetime import datetime, timedelta

# 1. Inicializar la API de Earth Engine
ee.Initialize()

# 2. Definir tu nuevo polígono
coordenadas = [
  [-4.811755842895575, 41.37409504652195],
  [-4.812917239399024, 41.37351537008459],
  [-4.811444593553215, 41.37225123589853],
  [-4.810167967851344, 41.37273024205082],
  [-4.811755842895575, 41.37409504652195]
]
aoi = ee.Geometry.Polygon(coordenadas)

# 3. Calcular rango de fechas dinámico (Últimos 7 días)
hoy = datetime.now()
hace_una_semana = hoy - timedelta(days=7)
fecha_inicio = hace_una_semana.strftime('%Y-%m-%d')
fecha_fin = hoy.strftime('%Y-%m-%d')

# 4. Función de procesamiento Sentinel-2 (NDVI)
def procesar_s2(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    img_norm = image.updateMask(mask).divide(10000)
    ndvi = img_norm.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return img_norm.addBands(ndvi).copyProperties(image, ['system:time_start'])

# 5. Cargar colección de la última semana
coleccion = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(aoi) \
    .filterDate(fecha_inicio, fecha_fin) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
    .map(procesar_s2)

# 6. Extraer los datos a memoria de manera síncrona (rápido para un solo polígono)
def extraer_datos(image):
    fecha = image.date().format('YYYY-MM-dd')
    valor_medio = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10
    ).get('NDVI')
    return ee.Feature(None, {'Fecha': fecha, 'NDVI': valor_medio})

# Obtenemos los resultados directamente en una lista de Python
datos_ee = coleccion.map(extraer_datos).filter(ee.Filter.notNull(['NDVI'])).getInfo()

# 7. Convertir a DataFrame y guardar a CSV directamente de forma local
features = [feat['properties'] for feat in datos_ee['features']]
if features:
    df = pd.DataFrame(features)
    # Guarda el nuevo archivo o puedes programarlo para que lo añada (append) a tu base de datos principal
    output_dir = os.path.join('datos', 'carralancha')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ndvi_recomendacion_semanal.csv')
    df.to_csv(output_path, index=False)
    print(f"Datos semanales descargados con éxito en {output_path}.")
else:
    print("No se encontraron imágenes sin nubes para esta semana.")