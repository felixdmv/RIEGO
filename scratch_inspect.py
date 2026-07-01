import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"
excel_files = glob.glob(os.path.join(folder, "*.xlsx"))

for f in sorted(excel_files):
    name = os.path.basename(f)
    try:
        xl = pd.ExcelFile(f)
        print(f"\nArchivo: {name}")
        print(f"Hojas: {xl.sheet_names}")
        # Leer primera hoja
        df = xl.parse(xl.sheet_names[0])
        print(f"Filas, Columnas: {df.shape}")
        print(f"Columnas: {list(df.columns)[:10]}...")
    except Exception as e:
        print(f"Error leyendo {name}: {e}")
