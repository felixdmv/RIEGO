import os
import re
import glob
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

dir_new = r"datos/canduela/GEE_Tesis_NuevaParcela"
tif_files = glob.glob(os.path.join(dir_new, "*.tif"))

records = []
for path in sorted(tif_files):
    filename = os.path.basename(path)
    m = re.search(r'(\d{4})-\d{2}-(\d{1,3})', filename)
    if m:
        try:
            date_val = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j")
            fecha_str = date_val.strftime("%Y-%m-%d")
            
            with rasterio.open(path) as src:
                b4 = src.read(3).astype(np.float32)
                b8 = src.read(7).astype(np.float32)
                
                # Exclude pixels where both bands are zero (nodata/background)
                valid_mask = (b4 > 1e-4) | (b8 > 1e-4)
                
                ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
                ndvi[~valid_mask] = np.nan
                mean_ndvi = np.nanmean(ndvi)
                
            records.append({
                "Fecha": fecha_str,
                "Valor": mean_ndvi
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

df = pd.DataFrame(records)
print(f"Computed NDVI for {len(df)} dates.")

# Interpolate any NaN values in the sequence
df['Valor'] = df['Valor'].interpolate(method='linear', limit_direction='both')
df['Valor'] = df['Valor'].bfill().ffill()

# Save to destination
output_csv = r"datos/canduela/satelite/Sentinel2_NDVI_Parcela_Real.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"Successfully saved updated NDVI data to {output_csv}.")
print("Top 10 rows:")
print(df.head(10))
print("Bottom 10 rows:")
print(df.tail(10))
