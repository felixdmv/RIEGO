import os
import re
import glob
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

dir_old = r"datos/canduela/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips"
csv_path = r"datos/canduela/satelite/Sentinel2_NDVI_Parcela_Real.csv"

df_csv = pd.read_csv(csv_path)
df_csv['Fecha'] = pd.to_datetime(df_csv['Fecha'])

# Parse TIF dates and calculate mean NDVI
tif_data = []
tif_files = glob.glob(os.path.join(dir_old, "*.tif"))
for path in tif_files:
    filename = os.path.basename(path)
    m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', filename)
    if m:
        try:
            date_val = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")
            with rasterio.open(path) as src:
                b4 = src.read(3).astype(np.float32)
                b8 = src.read(7).astype(np.float32)
                ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
                mean_ndvi = np.nanmean(ndvi)
                tif_data.append({"Fecha": pd.Timestamp(date_val), "Mean_NDVI_Calculated": mean_ndvi, "File": filename})
        except Exception as e:
            print(f"Error parsing/reading {filename}: {e}")

df_tif = pd.DataFrame(tif_data)
merged = pd.merge(df_csv, df_tif, on="Fecha", how="inner")
print(f"Merged count: {len(merged)}")
print("Sample comparison (CSV vs Calculated):")
print(merged.head(10)[["Fecha", "Valor", "Mean_NDVI_Calculated", "File"]])
print("\nMax difference:", np.abs(merged["Valor"] - merged["Mean_NDVI_Calculated"]).max())
