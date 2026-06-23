import os
import requests
import pandas as pd
import time
from datetime import date, timedelta

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"
ID_ESTACION = "2507Y"  # Rueda (Valladolid)

def fetch_aemet_data(fecha_inicio_str, fecha_fin_str, id_estacion):
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_inicio_str}T00:00:00UTC/fechafin/{fecha_fin_str}T23:59:59UTC/estacion/{id_estacion}"
    
    headers = {
        'cache-control': "no-cache",
        'api_key': API_KEY
    }
    
    print(f"Pidiendo datos a AEMET para la estación {id_estacion} desde {fecha_inicio_str} hasta {fecha_fin_str}...")
    try:
        response_inicial = requests.get(url_base, headers=headers)
        response_inicial.raise_for_status()
        
        json_inicial = response_inicial.json()
        
        if json_inicial.get("estado") == 200:
            url_datos = json_inicial["datos"]
            response_datos = requests.get(url_datos, headers=headers)
            response_datos.raise_for_status()
            
            datos_json = response_datos.json()
            df_aemet = pd.DataFrame(datos_json)
            return df_aemet
        else:
            print(f"[ERROR AEMET]: {json_inicial.get('descripcion')}")
            return None
    except Exception as e:
        print(f"Error descargando datos: {e}")
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
    
    for col_orig in columnas.keys():
        if col_orig not in df.columns:
            df[col_orig] = None
            
    df = df[list(columnas.keys())].rename(columns=columnas)
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    for col in ['Lluvia_AEMET', 'TempMedia_AEMET', 'TempMax_AEMET', 'TempMin_AEMET']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = df[col].str.replace('Ip', '0.0', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

if __name__ == "__main__":
    output_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "AEMET_Rueda_historico.csv")
    
    fecha_fin_total = date(2026, 6, 23)
    fecha_inicio_total = date(2022, 1, 1)

    lista_df_aemet = []
    fecha_chunk_inicio = fecha_inicio_total
    
    print("Iniciando descarga de datos de AEMET para Rueda...")

    while fecha_chunk_inicio < fecha_fin_total:
        fecha_chunk_fin = fecha_chunk_inicio + timedelta(days=150)
        if fecha_chunk_fin > fecha_fin_total:
            fecha_chunk_fin = fecha_fin_total
            
        fecha_inicio_str = fecha_chunk_inicio.strftime("%Y-%m-%d")
        fecha_fin_str = fecha_chunk_fin.strftime("%Y-%m-%d")

        df_bloque = fetch_aemet_data(fecha_inicio_str, fecha_fin_str, ID_ESTACION)
        
        if df_bloque is not None and not df_bloque.empty:
            lista_df_aemet.append(df_bloque)
        
        fecha_chunk_inicio = fecha_chunk_fin + timedelta(days=1)
        time.sleep(1.5)

    if lista_df_aemet:
        df_aemet_total = pd.concat(lista_df_aemet, ignore_index=True)
        df_aemet_limpio = limpiar_y_formatear_aemet(df_aemet_total)
        
        # Guardar en CSV
        df_aemet_limpio.to_csv(output_path, index=False)
        print(f"\n[OK] Datos de AEMET guardados en '{output_path}'")
        print(f"Filas guardadas: {len(df_aemet_limpio)}")
    else:
        print("[ERROR] No se pudieron descargar datos de AEMET")
