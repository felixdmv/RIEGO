import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
import math

def main():
    print("=====================================================================")
    print("INICIANDO PROCESO SEMANAL DE RECOMENDACIÓN DE RIEGO - CANDUELA SECTOR 2")
    print("=====================================================================")
    
    # 1. Descargar últimas mediciones climáticas de AEMET (Estación Sardón 2172Y)
    print("\n[Paso 1] Actualizando datos climáticos de AEMET (Estación Sardón de Duero)...")
    try:
        subprocess.run([sys.executable, "descargar_aemet_sardon.py"], check=True)
    except Exception as e:
        print(f"[AVISO] Error al actualizar AEMET, se utilizarán los datos locales existentes: {e}")
        
    # 2. Consolidar el Dataset Final (Cesens Excels, AEMET, Riegos, Satélites, FloraPulse)
    print("\n[Paso 2] Consolidando Dataset Final (Dataset_Final_S2.csv)...")
    try:
        subprocess.run([
            sys.executable, 
            "run_pipeline.py", 
            "--finca", "canduela", 
            "--sector", "2",
            "--days", "7",
            "--mode", "preprocess"
        ], check=True)
    except Exception as e:
        print(f"[ERROR] Error en el preprocesamiento de datos: {e}")
        return

    # 3. Generar informe de recomendación semanal para la fecha más reciente
    print("\n[Paso 3] Generando prescripción de riego semanal...")
    csv_path = os.path.join("datos", "canduela", "Dataset_Final_S2.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        # Obtener la última fecha disponible con datos
        ultima_fecha = df['Fecha'].max()
        ultima_fecha_str = ultima_fecha.strftime('%Y-%m-%d')
        print(f"Última fecha de datos disponible: {ultima_fecha_str}")
        
        try:
            subprocess.run([
                sys.executable, 
                "run_pipeline.py", 
                "--finca", "canduela", 
                "--sector", "2",
                "--days", "7",
                "--mode", "report", 
                "--date", ultima_fecha_str
            ], check=True)
            print(f"\n[OK] ¡Proceso completado exitosamente!")
            print(f"-> Se ha generado y guardado el PDF: recomendaciones/Prescripcion_S2_7d_{ultima_fecha_str}.pdf")
            
            # Imprimir resumen de la recomendación en consola (alineado con la lógica de run_pipeline.py)
            row = df[df['Fecha'] == ultima_fecha].iloc[0]
            kc = 0.4 + (row['Sat_NDVI'] * 0.8)
            
            # Calcular ETc representativo (usando la media de los 3 días anteriores si el último día está incompleto)
            df_temp = df[df['Fecha'] <= ultima_fecha].copy()
            es_hoy = (ultima_fecha.date() == datetime.today().date())
            mes = ultima_fecha.month
            es_epoca_riego = 4 <= mes <= 10
            es_incompleto = (es_hoy and datetime.now().hour < 22) or (es_epoca_riego and row['ET0_Calculada'] < 1.5)
            
            if es_incompleto and len(df_temp) > 1:
                etc_rep = df_temp.iloc[:-1]['ETc_Sector2'].tail(3).mean()
                if pd.isna(etc_rep):
                    etc_rep = row['ETc_Sector2']
            else:
                etc_rep = df_temp['ETc_Sector2'].tail(3).mean()
                if pd.isna(etc_rep):
                    etc_rep = row['ETc_Sector2']
            
            etc = etc_rep
            dosis_ideal_base = etc * 7 - row['Lluvia_Prevista_7d']
            
            # Estrategia RDC al 70%
            dosis_rec = max(0, dosis_ideal_base * 0.70)
            presc = "Estrategia de Riego Deficitario Controlado (RDC) al 70%. Se prescribe riego de mantenimiento para sostener el vigor y el desarrollo del fruto."
            
            # Area is 16698.54 m2 and precipitation rate is 1.09 mm/h for Sector 2
            volumen_m3 = (dosis_rec * 16698.54) / 1000.0
            irr_hours = dosis_rec / 1.09

            if dosis_rec <= 0:
                irr_hours_str = "0 minutos"
                turns = 0
                hours_per_turn_str = "0 minutos"
                days_str = "No se requiere riego"
            else:
                turns = min(7, max(1, int(math.ceil(irr_hours / 6.0))))
                hours_per_turn = irr_hours / turns
                
                def format_duration(h):
                    hrs = int(h)
                    mins = int(round((h - hrs) * 60))
                    if mins == 60:
                        hrs += 1
                        mins = 0
                    parts = []
                    if hrs > 0:
                        parts.append(f"{hrs} {'hora' if hrs == 1 else 'horas'}")
                    if mins > 0:
                        parts.append(f"{mins} {'minuto' if mins == 1 else 'minutos'}")
                    return " y ".join(parts) if parts else "0 minutos"

                irr_hours_str = format_duration(irr_hours)
                hours_per_turn_str = format_duration(hours_per_turn)

                if turns == 1:
                    days_list = ["Martes"]
                elif turns == 2:
                    days_list = ["Martes", "Viernes"]
                elif turns == 3:
                    days_list = ["Martes", "Jueves", "Sábado"]
                elif turns == 4:
                    days_list = ["Martes", "Jueves", "Sábado", "Lunes"]
                elif turns == 5:
                    days_list = ["Martes", "Miércoles", "Viernes", "Sábado", "Lunes"]
                elif turns == 6:
                    days_list = ["Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                else:
                    days_list = ["Todos los días (de Martes a Lunes)"]
                
                if len(days_list) == 1:
                    days_str = days_list[0]
                elif len(days_list) > 1 and days_list[0] != "Todos los días (de Martes a Lunes)":
                    days_str = ", ".join(days_list[:-1]) + " y " + days_list[-1]
                else:
                    days_str = days_list[0]

            print(f"\n=======================================================")
            print(f" RESUMEN DE LA PRESCRIPCIÓN ({ultima_fecha_str})")
            print(f"=======================================================")
            print(f" * NDVI Parcela: {row['Sat_NDVI']:.3f}")
            print(f" * Kc estimado: {kc:.2f}")
            print(f" * ETc diaria calculada (rep): {etc:.2f} mm/día")
            print(f" * Previsión de lluvia (7 días): {row['Lluvia_Prevista_7d']:.1f} mm")
            print(f" * Humedad del suelo (40cm): {row['FP_Humedad_40cm']:.1f}%")
            print(f"-------------------------------------------------------")
            print(f" * DOSIS RECOMENDADA (7 días): {dosis_rec:.1f} mm")
            print(f" * VOLUMEN TOTAL RECOMENDADO: {volumen_m3:.1f} m³")
            print(f" * TIEMPO DE RIEGO ESTIMADO: {irr_hours_str} (a 1.09 mm/h)")
            if turns > 0:
                print(f" * PROGRAMACIÓN SUGERIDA: {turns} {'turno' if turns == 1 else 'turnos'} de {hours_per_turn_str} cada uno")
                print(f" * DÍAS SUGERIDOS: {days_str} (siempre y cuando la meteorología lo permita)")
            else:
                print(f" * PROGRAMACIÓN SUGERIDA: No se requiere riego")
            print(f" * ESTRATEGIA: {presc}")
            print(f"=======================================================")
            
        except Exception as e:
            print(f"[ERROR] Error al generar el PDF de prescripción: {e}")
    else:
        print("[ERROR] No se pudo encontrar el Dataset consolidado final.")

if __name__ == "__main__":
    main()
