import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"

clima_files = glob.glob(os.path.join(folder, "CLIMA*.xlsx"))
crop_files = glob.glob(os.path.join(folder, "CARRALANCHA*.xlsx"))

print("CLIMA FILES DATES:")
for f in sorted(clima_files):
    df = pd.read_excel(f, usecols=['Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"  {os.path.basename(f)}: {df['Fecha'].min()} to {df['Fecha'].max()} (rows: {len(df)})")

print("\nCROP FILES DATES:")
for f in sorted(crop_files):
    df = pd.read_excel(f, usecols=['Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    print(f"  {os.path.basename(f)}: {df['Fecha'].min()} to {df['Fecha'].max()} (rows: {len(df)})")
