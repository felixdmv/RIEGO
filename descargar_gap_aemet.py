import os
import requests
import pandas as pd
import time
from datetime import date

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"
ID_ESTACION = "2507Y"
csv_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\AEMET_Rueda_historico.csv"

def fetch_aemet_data(fecha_inicio_str, fecha_fin_str, id_estacion):
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_inicio_str}T00:00:00UTC/fechafin/{fecha_fin_str}T23:59:59UTC/estacion/{id_estacion}"
    headers = {'cache-control': "no-cache", 'api_key': API_KEY}
    try:
        response = requests.get(url_base, headers=headers)
        if response.status_code == 200:
            js = response.json()
            if js.get("estado") == 200:
                url_datos = js["datos"]
                res_datos = requests.get(url_datos, headers=headers)
                return pd.DataFrame(res_datos.json())
        print(f"Error HTTP o estado no 200: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    return None

def limpiar_y_formatear_aemet(df):
    if df is None or df.empty:
        return pd.DataFrame()
    columnas = {
        'fecha': 'Fecha',
        'prec': 'Lluvia_AEMET',
        'tmed': 'TempMedia_AEMET',
        'tmax': 'TempMax_AEMET',
        'tmin': 'TempMin_AEMET'
    }
    df = df[list(columnas.keys())].rename(columns=columnas)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    for col in ['Lluvia_AEMET', 'TempMedia_AEMET', 'TempMax_AEMET', 'TempMin_AEMET']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = df[col].str.replace('Ip', '0.0', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if __name__ == "__main__":
    # Cargar datos existentes
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old['Fecha'] = pd.to_datetime(df_old['Fecha'])
    else:
        df_old = pd.DataFrame(columns=['Fecha', 'Lluvia_AEMET', 'TempMedia_AEMET', 'TempMax_AEMET', 'TempMin_AEMET'])

    print(f"Descargando el bloque faltante (2022-01-01 -> 2022-05-31) para Rueda...")
    # Esperar un momento por si acaso para evitar 429
    time.sleep(2)
    
    df_missing = fetch_aemet_data("2022-01-01", "2022-05-31", ID_ESTACION)
    if df_missing is not None and not df_missing.empty:
        df_missing_cleaned = limpiar_y_formatear_aemet(df_missing)
        df_new = pd.concat([df_missing_cleaned, df_old], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)
        df_new.to_csv(csv_path, index=False)
        print(f"[OK] Datos unificados. Nuevas filas totales en el archivo: {len(df_new)}")
    else:
        print("[ERROR] No se pudo descargar el bloque faltante.")
