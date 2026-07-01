import pandas as pd
import glob
import os

folder = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos\carralancha"

clima_file = os.path.join(folder, "CLIMA CARRALANCHA 2022 ENERO-JULIO.xlsx")
crop_file = os.path.join(folder, "CARRALANCHA ALMENDRO V4 2022 ENERO-JULIO.xlsx")

if os.path.exists(clima_file):
    df_clima = pd.read_excel(clima_file, sheet_name=0, nrows=2)
    print("CLIMA COLUMNS:")
    print(list(df_clima.columns))

if os.path.exists(crop_file):
    df_crop = pd.read_excel(crop_file, sheet_name=0, nrows=2)
    print("\nCROP (ALMENDRO) COLUMNS:")
    print(list(df_crop.columns))
