import os
import requests
import pandas as pd
import time
from datetime import datetime, date, timedelta

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"
ID_ESTACION = "2172Y"  # Sardón de Duero

def fetch_aemet_data_with_retry(fecha_inicio_str, fecha_fin_str, id_estacion, max_retries=5, backoff_base=5):
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_inicio_str}T00:00:00UTC/fechafin/{fecha_fin_str}T23:59:59UTC/estacion/{id_estacion}"
    headers = {
        'cache-control': "no-cache",
        'api_key': API_KEY
    }
    
    for attempt in range(max_retries):
        print(f"Pidiendo datos a AEMET ({fecha_inicio_str} a {fecha_fin_str}). Intento {attempt + 1}/{max_retries}...")
        try:
            response_inicial = requests.get(url_base, headers=headers, timeout=30)
            if response_inicial.status_code == 429:
                sleep_time = backoff_base * (2 ** attempt)
                print(f"[AVISO AEMET]: Código 429 (Too Many Requests). Esperando {sleep_time} segundos antes de reintentar...")
                time.sleep(sleep_time)
                continue
                
            response_inicial.raise_for_status()
            json_inicial = response_inicial.json()
            
            if json_inicial.get("estado") == 200:
                url_datos = json_inicial["datos"]
                time.sleep(1.0)
                response_datos = requests.get(url_datos, headers=headers, timeout=30)
                response_datos.raise_for_status()
                datos_json = response_datos.json()
                df_aemet = pd.DataFrame(datos_json)
                return df_aemet
            elif json_inicial.get("estado") == 404:
                print(f"[AVISO AEMET]: Sin datos encontrados (404) para el rango {fecha_inicio_str} a {fecha_fin_str}.")
                return None
            else:
                print(f"[ERROR AEMET]: {json_inicial.get('descripcion')}")
                return None
        except Exception as e:
            print(f"Error en intento {attempt + 1}: {e}")
            time.sleep(backoff_base)
            
    print(f"[ERROR]: Fallaron todos los intentos para el rango {fecha_inicio_str} a {fecha_fin_str}.")
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

def main():
    output_dir = os.path.join("datos", "canduela")
    output_path = os.path.join(output_dir, "AEMET_Sardon_historico.csv")
    
    if os.path.exists(output_path):
        print(f"Cargando archivo histórico existente desde {output_path}...")
        df_existente = pd.read_csv(output_path)
        df_existente['Fecha'] = pd.to_datetime(df_existente['Fecha'])
    else:
        print("No se encontró histórico anterior. Creando uno nuevo...")
        df_existente = pd.DataFrame(columns=['Fecha', 'Lluvia_AEMET', 'TempMedia_AEMET', 'TempMax_AEMET', 'TempMin_AEMET'])
        df_existente['Fecha'] = pd.to_datetime(df_existente['Fecha'])
        
    fecha_inicio_rango = pd.to_datetime('2022-01-01')
    fecha_fin_rango = pd.Timestamp(date.today() - timedelta(days=3))
    
    print(f"Buscando huecos de datos desde {fecha_inicio_rango.strftime('%Y-%m-%d')} hasta {fecha_fin_rango.strftime('%Y-%m-%d')}...")
    
    all_dates = pd.date_range(start=fecha_inicio_rango, end=fecha_fin_rango)
    missing_dates = all_dates.difference(df_existente['Fecha'])
    
    if len(missing_dates) == 0:
        print("[OK] No hay fechas faltantes en el histórico. ¡Está completo!")
        return

    print(f"Se detectaron {len(missing_dates)} días sin registrar.")
    
    bloques = []
    if len(missing_dates) > 0:
        s = pd.Series(missing_dates)
        gps = s.groupby(s.diff().ne(pd.offsets.Day(1)).cumsum())
        for _, group in gps:
            bloques.append((group.min(), group.max()))
            
    print(f"Se identificaron {len(bloques)} bloques continuos de datos faltantes:")
    for b in bloques:
        print(f"  - Desde {b[0].strftime('%Y-%m-%d')} hasta {b[1].strftime('%Y-%m-%d')} ({ (b[1]-b[0]).days + 1 } días)")

    nuevos_datos = []
    
    for b_ini, b_fin in bloques:
        sub_ini = b_ini
        while sub_ini <= b_fin:
            sub_fin = sub_ini + timedelta(days=150)
            if sub_fin > b_fin:
                sub_fin = b_fin
                
            df_bloque_raw = fetch_aemet_data_with_retry(
                sub_ini.strftime("%Y-%m-%d"), 
                sub_fin.strftime("%Y-%m-%d"), 
                ID_ESTACION
            )
            
            if df_bloque_raw is not None and not df_bloque_raw.empty:
                df_bloque_limpio = limpiar_y_formatear_aemet(df_bloque_raw)
                if not df_bloque_limpio.empty:
                    nuevos_datos.append(df_bloque_limpio)
                    print(f"  -> Descargadas {len(df_bloque_limpio)} filas para el sub-rango.")
            
            sub_ini = sub_fin + timedelta(days=1)
            time.sleep(3.5)
            
    if nuevos_datos:
        df_nuevos = pd.concat(nuevos_datos, ignore_index=True)
        df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['Fecha'], keep='last')
        df_final = df_final.sort_values('Fecha').reset_index(drop=True)
        
        df_final.to_csv(output_path, index=False)
        print(f"\n[OK] ¡Histórico actualizado! Guardado en '{output_path}'")
        print(f"Filas totales ahora: {len(df_final)}")
        
        missing_after = all_dates.difference(df_final['Fecha'])
        print(f"Huecos restantes: {len(missing_after)} días.")
    else:
        print("\nNo se pudieron descargar nuevos datos.")

if __name__ == "__main__":
    main()
