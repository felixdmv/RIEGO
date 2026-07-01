import requests
import math

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"
url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"

headers = {
    'cache-control': "no-cache",
    'api_key': API_KEY
}

# Coordenadas de la finca en Carralancha (de extraer_datos_satelite.py)
# [[-4.811755842895575, 41.37409504652195], ...]
lat_finca = 41.3735
lon_finca = -4.8115

def haversine(lat1, lon1, lat2, lon2):
    # Radio de la Tierra en km
    R = 6371.0
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

print("Pidiendo inventario de estaciones...")
r = requests.get(url, headers=headers)
if r.status_code == 200:
    res = r.json()
    if res.get("estado") == 200:
        url_datos = res["datos"]
        r2 = requests.get(url_datos, headers=headers)
        estaciones = r2.json()
        
        # Calcular distancias
        cercanas = []
        for est in estaciones:
            # Latitud y longitud vienen en formatos como "412356N" o "045612W" (grados, minutos, segundos)
            # Vamos a convertirlos o buscar Valladolid / Medina del Campo
            lat_str = est.get("latitud", "")
            lon_str = est.get("longitud", "")
            
            # Convertir latitud (ej: 412356N)
            # DDMMSSN -> grados + minutos/60 + segundos/3600
            try:
                lat_deg = float(lat_str[:2]) + float(lat_str[2:4])/60.0 + float(lat_str[4:6])/3600.0
                if lat_str[-1] == 'S':
                    lat_deg = -lat_deg
                
                # Convertir longitud (ej: 045612W)
                # DDDMMSSW -> grados + minutos/60 + segundos/3600
                lon_deg = float(lon_str[:2]) + float(lon_str[2:4])/60.0 + float(lon_str[4:6])/3600.0
                if lon_str[-1] == 'W':
                    lon_deg = -lon_deg
                
                dist = haversine(lat_finca, lon_finca, lat_deg, lon_deg)
                cercanas.append((dist, est.get("indicativo"), est.get("nombre"), est.get("provincia"), lat_deg, lon_deg))
            except Exception as e:
                # Si falla la conversión, ignorar o imprimir
                pass
        
        cercanas.sort()
        print("\nEstaciones más cercanas a la finca en Pozaldez:")
        for dist, ind, nom, prov, lat, lon in cercanas[:15]:
            print(f"- {nom} ({prov}): {dist:.2f} km, indicativo={ind}, lat={lat:.4f}, lon={lon:.4f}")
    else:
        print("Error en respuesta:", res.get("descripcion"))
else:
    print("Error HTTP:", r.status_code)
