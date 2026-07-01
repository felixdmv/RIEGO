import rasterio
import os
import glob
import numpy as np

dir_old = r"datos/canduela/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips"
dir_new = r"datos/canduela/GEE_Tesis_NuevaParcela"

file_old = glob.glob(os.path.join(dir_old, "*.tif"))[0]
file_new = glob.glob(os.path.join(dir_new, "*.tif"))[0]

for name, path in [("Old", file_old), ("New", file_new)]:
    print(f"=== {name} File: {os.path.basename(path)} ===")
    with rasterio.open(path) as src:
        print(f"  Bands count: {src.count}")
        print(f"  Width: {src.width}, Height: {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        for i in range(1, src.count + 1):
            band_data = src.read(i)
            print(f"    Band {i}: min={band_data.min():.4f}, max={band_data.max():.4f}, mean={band_data.mean():.4f}, nan_count={np.isnan(band_data).sum()}")
        # Let's check if we can compute NDVI
        try:
            b4 = src.read(3).astype(np.float32)
            b8 = src.read(7).astype(np.float32)
            ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
            print(f"  NDVI: min={np.nanmin(ndvi):.4f}, max={np.nanmax(ndvi):.4f}, mean={np.nanmean(ndvi):.4f}")
        except Exception as e:
            print(f"  Error calculating NDVI: {e}")
