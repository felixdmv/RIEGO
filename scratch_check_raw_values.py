import pandas as pd
import glob
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')

excel_files = glob.glob(os.path.join(datos_path, 'florapulse', "*2024*.xlsx"))

dfs = []
for f in excel_files:
    try:
        xl = pd.ExcelFile(f)
        df = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df.columns if 'fecha' in str(c).lower()][0]
        pot_raw_col = [c for c in df.columns if 'potencial' in str(c).lower() and 'bruto' in str(c).lower()][0]
        
        df = df.rename(columns={fecha_col: 'Fecha', pot_raw_col: 'Potencial_Raw'})
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        df['Potencial_Raw'] = pd.to_numeric(df['Potencial_Raw'], errors='coerce')
        df = df.dropna(subset=['Potencial_Raw'])
        
        dfs.append(df[['Fecha', 'Potencial_Raw']])
    except Exception as e:
        print(f"Error {f}: {e}")

if dfs:
    df_fp = pd.concat(dfs, ignore_index=True).sort_values('Fecha')
    df_fp_daily = df_fp.set_index('Fecha').resample('D').min().reset_index()
    
    # Filter Summer 2024
    df_summer = df_fp_daily[(df_fp_daily['Fecha'] >= '2024-06-01') & (df_fp_daily['Fecha'] <= '2024-09-30')]
    print("\n--- Daily Aggregated Potential (En Bruto) for Summer 2024 ---")
    print(df_summer.describe())
    print("\nSample of daily values (first 20 rows of Summer 2024):")
    print(df_summer.head(20))
    print("\nChecking dates with high stress (potential < -10 bar):")
    print(df_summer[df_summer['Potencial_Raw'] < -10.0].head(10))
