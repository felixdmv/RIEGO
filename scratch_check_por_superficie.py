import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
crop_files = glob.glob(os.path.join(folder, "CARRALANCHA*.xlsx"))

for f in sorted(crop_files):
    name = os.path.basename(f)
    try:
        df = pd.read_excel(f)
        for col in ['Riego (por superficie)', 'Riego diario (por superficie)']:
            if col in df.columns:
                nz = df[df[col] > 0]
                if not nz.empty:
                    print(f"\n{name} - {col} tiene valores > 0:")
                    print(nz[['Fecha', col]].head(5))
    except Exception as e:
        print(f"Error reading {name}: {e}")
