import requests
import pandas as pd

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"

def fetch_aemet_data(fecha_inicio_str, fecha_fin_str, id_estacion):
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_inicio_str}T00:00:00UTC/fechafin/{fecha_fin_str}T23:59:59UTC/estacion/{id_estacion}"
    headers = {'cache-control': "no-cache", 'api_key': API_KEY}
    response_inicial = requests.get(url_base, headers=headers)
    if response_inicial.status_code == 200:
        json_inicial = response_inicial.json()
        if json_inicial.get("estado") == 200:
            url_datos = json_inicial["datos"]
            response_datos = requests.get(url_datos, headers=headers)
            return pd.DataFrame(response_datos.json())
    return None

df_olmedo = fetch_aemet_data("2024-06-01", "2024-06-10", "2503X")
print("Olmedo 2503X columns and sample:")
if df_olmedo is not None:
    print(df_olmedo.columns)
    print(df_olmedo[['fecha', 'prec', 'tmed', 'tmax', 'tmin']].head(5))
else:
    print("Failed to download from Olmedo")

df_rueda = fetch_aemet_data("2024-06-01", "2024-06-10", "2507Y")
print("\nRueda 2507Y columns and sample:")
if df_rueda is not None:
    print(df_rueda.columns)
    print(df_rueda[['fecha', 'prec', 'tmed', 'tmax', 'tmin']].head(5))
else:
    print("Failed to download from Rueda")
