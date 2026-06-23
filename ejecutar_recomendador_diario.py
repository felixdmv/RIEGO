import os
import subprocess
import pandas as pd
from datetime import date, datetime

def main():
    print("=====================================================================")
    print("INICIANDO PROCESO DIARIO DE RECOMENDACIÓN DE RIEGO - CARRALANCHA V4")
    print("=====================================================================")
    
    # 1. Descargar últimas mediciones climáticas de AEMET (Estación Rueda)
    print("\n[Paso 1] Actualizando datos climáticos de AEMET (Estación Rueda)...")
    try:
        # Ejecutar el script descargar_aemet_rueda.py para actualizar datos hasta hoy
        subprocess.run(["C:\\Users\\fdemiguel\\AppData\\Local\\anaconda3\\envs\\agro_ft\\python.exe", "descargar_aemet_rueda.py"], check=True)
    except Exception as e:
        print(f"[AVISO] Error al actualizar AEMET, se utilizarán los datos locales existentes: {e}")
        
    # 2. Descargar últimos índices satelitales (NDVI, Radar, LST)
    print("\n[Paso 2] Actualizando índices satelitales desde Earth Engine...")
    try:
        subprocess.run(["C:\\Users\\fdemiguel\\AppData\Local\\anaconda3\\envs\\agro_ft\\python.exe", "extraer_datos_satelite_completo.py"], check=True)
    except Exception as e:
        print(f"[AVISO] Error al actualizar datos satelitales: {e}")

    # 3. Unificar los Excels de sensores y riegos de la finca
    print("\n[Paso 3] Unificando y convirtiendo excels de sensores a CSV...")
    try:
        subprocess.run(["C:\\Users\\fdemiguel\\AppData\\Local\\anaconda3\\envs\\agro_ft\\python.exe", "unificar_excels_carralancha.py"], check=True)
    except Exception as e:
        print(f"[ERROR] Error al unificar excels: {e}")
        return

    # 4. Consolidar el Dataset Final e interpolar
    print("\n[Paso 4] Consolidando Dataset Final (Dataset_Final_V4.csv)...")
    try:
        # Ejecutar preprocesamiento en run_pipeline.py
        subprocess.run([
            "C:\\Users\\fdemiguel\\AppData\\Local\\anaconda3\envs\\agro_ft\\python.exe", 
            "run_pipeline.py", 
            "--finca", "carralancha", 
            "--mode", "preprocess"
        ], check=True)
    except Exception as e:
        print(f"[ERROR] Error en el preprocesamiento de datos: {e}")
        return

    # 5. Generar informe de recomendación semanal para la fecha más reciente
    print("\n[Paso 5] Generando prescripción de riego semanal...")
    csv_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha\Dataset_Final_V4.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        # Obtener la última fecha disponible con datos
        ultima_fecha = df['Fecha'].max()
        ultima_fecha_str = ultima_fecha.strftime('%Y-%m-%d')
        print(f"Última fecha de datos disponible: {ultima_fecha_str}")
        
        try:
            subprocess.run([
                "C:\\Users\\fdemiguel\\AppData\\Local\\anaconda3\\envs\\agro_ft\\python.exe", 
                "run_pipeline.py", 
                "--finca", "carralancha", 
                "--mode", "report", 
                "--date", ultima_fecha_str
            ], check=True)
            print(f"\n[OK] ¡Proceso completado exitosamente!")
            print(f"-> Se ha generado y guardado el PDF: Prescripcion_V4_{ultima_fecha_str}.pdf")
            
            # Imprimir resumen de la recomendación en consola
            # Leer el último día de datos consolidado para mostrar la prescripción
            row = df[df['Fecha'] == ultima_fecha].iloc[0]
            kc = 0.4 + (row['Sat_NDVI'] * 0.8)
            etc = row['ET0_Calculada'] * kc
            dosis_ideal_base = etc * 3 - row['Lluvia_Prevista_3d']
            pot = row['FP_Potencial_Min']
            
            if pot >= -8.0:
                dosis_rec = max(0, dosis_ideal_base * 0.70)
                presc = "Suelo húmedo y potencial óptimo. Se prescribe riego de mantenimiento para mantener el nivel actual de vigor."
            elif -12.0 <= pot < -8.0:
                dosis_rec = max(0, dosis_ideal_base * 0.60)
                presc = "Estrés leve favorable para el cultivo en esta etapa. Mantener estrategia de RDI con dosis controlada."
            elif -16.0 <= pot < -12.0:
                dosis_rec = max(0, dosis_ideal_base * 0.85 + 3.0)
                presc = "Se detecta estrés moderado en FloraPulse. Incrementar dosis de riego para recuperar el potencial hídrico foliar."
            else:
                dosis_rec = max(0, dosis_ideal_base * 1.00 + 8.0)
                presc = "ALERTA CRÍTICA: Potencial hídrico en zona de peligro. Aplicar riego de choque de forma inmediata para restaurar turgencia celular."
            
            volumen_m3 = (dosis_rec * 22357.87) / 1000.0
            irr_hours = dosis_rec / 1.2
            print(f"\n=======================================================")
            print(f" RESUMEN DE LA PRESCRIPCIÓN ({ultima_fecha_str})")
            print(f"=======================================================")
            print(f" * NDVI Parcela: {row['Sat_NDVI']:.3f}")
            print(f" * Kc estimado: {kc:.2f}")
            print(f" * ETc diaria calculada: {etc:.2f} mm/día")
            print(f" * Previsión de lluvia (3 días): {row['Lluvia_Prevista_3d']:.1f} mm")
            print(f" * Potencial hídrico foliar: {pot:.1f} bar")
            print(f"-------------------------------------------------------")
            print(f" * DOSIS RECOMENDADA (3 días): {dosis_rec:.1f} mm")
            print(f" * VOLUMEN TOTAL RECOMENDADO: {volumen_m3:.1f} m³")
            print(f" * TIEMPO DE RIEGO ESTIMADO: {irr_hours:.1f} horas (a 1.2 mm/h)")
            print(f" * ESTRATEGIA: {presc}")
            print(f"=======================================================")
            
        except Exception as e:
            print(f"[ERROR] Error al generar el PDF de prescripción: {e}")
    else:
        print("[ERROR] No se pudo encontrar el Dataset consolidado final.")

if __name__ == "__main__":
    main()
