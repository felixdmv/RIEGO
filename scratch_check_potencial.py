import pandas as pd
import glob
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')

excel_files = glob.glob(os.path.join(datos_path, 'florapulse', "*.xlsx"))

dfs = []
for f in excel_files:
    try:
        xl = pd.ExcelFile(f)
        df = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df.columns if 'fecha' in str(c).lower()][0]
        df = df.rename(columns={fecha_col: 'Fecha'})
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        # Check potential columns
        pot_cols = [c for c in df.columns if 'potencial' in str(c).lower() and 'almendra' in str(c).lower()]
        for c in pot_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        cols_to_keep = ['Fecha'] + pot_cols
        df = df[cols_to_keep]
        dfs.append(df)
    except Exception as e:
        print(f"Error {os.path.basename(f)}: {e}")

if dfs:
    df_pot = pd.concat(dfs, ignore_index=True)
    df_pot = df_pot.dropna()
    print("Combined non-null rows:", len(df_pot))
    print("Columns:", df_pot.columns.tolist())
    print("\nSummary statistics:")
    print(df_pot.describe())
    print("\nFirst 10 non-null values:")
    print(df_pot.head(10))
