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
        
        pot_cols = [c for c in df.columns if 'potencial' in str(c).lower()]
        for c in pot_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        df = df[['Fecha'] + pot_cols]
        dfs.append(df)
    except Exception as e:
        pass

if dfs:
    df_pot = pd.concat(dfs, ignore_index=True)
    print("Columns:", df_pot.columns.tolist())
    print("\nDescribe summary for all potential columns:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_pot.describe())
