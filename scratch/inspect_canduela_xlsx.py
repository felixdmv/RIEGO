import pandas as pd
import os

def inspect_file(p):
    print(f"\n=== INSPECTING {os.path.basename(p)} ===")
    try:
        xl = pd.ExcelFile(p)
        print(f"Sheets: {xl.sheet_names}")
        df = xl.parse(xl.sheet_names[0])
        print(f"Shape: {df.shape}")
        print("Columns:")
        for col in df.columns:
            print(f"  - {col}")
        print("First 3 rows:")
        print(df.head(3))
        # check date col
        fcol = [c for c in df.columns if 'fecha' in str(c).lower() or 'comienzo' in str(c).lower() or 'inicio' in str(c).lower()]
        if fcol:
            df['Parsed'] = pd.to_datetime(df[fcol[0]], errors='coerce')
            print(f"Date range: {df['Parsed'].min()} to {df['Parsed'].max()}")
    except Exception as e:
        print(f"Error: {e}")

inspect_file('datos/canduela/CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx')
inspect_file('C:/Users/felix/Downloads/Datos estaciones Cesens (1).xlsx')
