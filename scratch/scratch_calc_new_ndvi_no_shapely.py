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
                # Valid pixels (not background/nodata)
                valid_mask = (b4 > 1e-4) | (b8 > 1e-4)
                
                ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
                ndvi[~valid_mask] = np.nan
                mean_ndvi = np.nanmean(ndvi)
                
            records.append({
                "Fecha": fecha_str,
                "Mean_NDVI": mean_ndvi,
                "File": filename
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

df_res = pd.DataFrame(records)
print(f"Processed {len(df_res)} files.")
print("Sample of calculated NDVI:")
print(df_res.head(15))
print("\nOverall Mean NDVI:", df_res["Mean_NDVI"].mean())
print("Overall Min NDVI:", df_res["Mean_NDVI"].min())
print("Overall Max NDVI:", df_res["Mean_NDVI"].max())
