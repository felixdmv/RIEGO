import pandas as pd
import glob
import os
import numpy as np

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')

excel_files = glob.glob(os.path.join(datos_path, 'florapulse', "*.xlsx"))

dfs = []
for f in excel_files:
    try:
        xl = pd.ExcelFile(f)
        df = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df.columns if 'fecha' in str(c).lower()][0]
        # Rename the date column to 'Fecha' to make it consistent
        df = df.rename(columns={fecha_col: 'Fecha'})
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        # Select only numeric columns and Fecha
        cols_to_keep = ['Fecha']
        for c in df.columns:
            if c != 'Fecha':
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                    cols_to_keep.append(c)
                except:
                    pass
        df = df[cols_to_keep]
        dfs.append(df)
        print(f"Loaded {os.path.basename(f)}: rows={len(df)}, columns={len(df.columns)}")
    except Exception as e:
        print(f"Error loading {os.path.basename(f)}: {e}")

if dfs:
    # We want to concatenate, but columns might slightly differ between files.
    # Let's align them.
    df_fp = pd.concat(dfs, ignore_index=True)
    df_fp = df_fp.sort_values('Fecha')
    print("\nCombined Florapulse shape:", df_fp.shape)
    print(f"Date range: {df_fp['Fecha'].min()} to {df_fp['Fecha'].max()}")
    
    # Aggregate daily
    df_fp_daily = df_fp.set_index('Fecha').resample('D').mean()
    print("Daily aggregated shape:", df_fp_daily.shape)
    
    # Calculate non-null percentage
    non_null_pct = df_fp_daily.notnull().mean() * 100
    print("\nNon-null percentage in daily data (top 25):")
    sorted_pct = non_null_pct.sort_values(ascending=False)
    for col, pct in sorted_pct.items():
        print(f"  {col}: {pct:.2f}% (Non-null count: {df_fp_daily[col].notnull().sum()})")
