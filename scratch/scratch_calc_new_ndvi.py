import os
import re
import glob
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from datetime import datetime
from shapely.geometry import Polygon

coords_s2 = [
    [-4.577624452593563, 41.814040723556154],
    [-4.577847075941799, 41.813829819817485],
    [-4.578032148363826, 41.813547947385764],
    [-4.578068697003681, 41.81318036452114],
    [-4.578187960151398, 41.81304849397068],
    [-4.5787659761941075, 41.81318443366328],
    [-4.5789922247252175, 41.8128168929197],
    [-4.57900161245677, 41.812716936717905],
    [-4.578906394036741, 41.81257571549979],
    [-4.579008317979307, 41.81245576758731],
    [-4.579288608821363, 41.81226179164462],
    [-4.579359687360258, 41.81213584571337],
    [-4.579383827241392, 41.81202789186088],
    [-4.579473681243391, 41.811804986571325],
    [-4.579570240767927, 41.81157808218388],
    [-4.579650509117174, 41.81137531452619],
    [-4.579696106670427, 41.811228375548794],
    [-4.579764503000307, 41.810963484704025],
    [-4.579886543510485, 41.81064661383948],
    [-4.580094414709139, 41.81020878968733],
    [-4.580137330053377, 41.810058849221306],
    [-4.580166834352541, 41.80985093119436],
    [-4.579960304258394, 41.80978995510655],
    [-4.579854357002306, 41.80994689344449],
    [-4.579501646516848, 41.80984693276414],
    [-4.57937424158864, 41.81008084051163],
    [-4.5798677680473805, 41.81022478331631],
    [-4.579641121385622, 41.81058863729716],
    [-4.579350874669361, 41.811062291391984],
    [-4.578583863489332, 41.81230090431014],
    [-4.5780997247621436, 41.813084561839695],
    [-4.577533597319558, 41.813997981766136],
    [-4.577624452593563, 41.814040723556154]
]

# We need the polygon in UTM coordinates
from rasterio.warp import transform as warp_transform
lons = [c[0] for c in coords_s2]
lats = [c[1] for c in coords_s2]
xs, ys = warp_transform('EPSG:4326', 'EPSG:32630', lons, lats)
poly_utm = Polygon(zip(xs, ys))
geojson_poly = [poly_utm.__geo_interface__]

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
                # 1. Compute NDVI for the entire chip
                b4_full = src.read(3).astype(np.float32)
                b8_full = src.read(7).astype(np.float32)
                ndvi_full = (b8_full - b4_full) / (b8_full + b4_full + 1e-10)
                mean_ndvi_full = np.nanmean(ndvi_full)
                
                # 2. Compute NDVI masked by Sector 2 polygon
                try:
                    # rasterio.mask expects list of GeoJSON-like dicts
                    out_image, out_transform = rasterio.mask.mask(src, geojson_poly, crop=True)
                    # out_image has shape (bands, height, width)
                    # band 3 is index 2, band 7 is index 6
                    b4_masked = out_image[2].astype(np.float32)
                    b8_masked = out_image[6].astype(np.float32)
                    ndvi_masked = (b8_masked - b4_masked) / (b8_masked + b4_masked + 1e-10)
                    
                    # exclude nodata values (which are usually 0 in masked area, check mask or values)
                    # wait, rasterio mask fills outside areas with src.nodata (usually 0 or nan)
                    # Let's mask values where b4_masked and b8_masked are exactly 0
                    mask_nodata = (b4_masked == 0) & (b8_masked == 0)
                    ndvi_masked[mask_nodata] = np.nan
                    mean_ndvi_masked = np.nanmean(ndvi_masked)
                except Exception as mask_err:
                    print(f"Mask error for {filename}: {mask_err}")
                    mean_ndvi_masked = mean_ndvi_full
                    
            records.append({
                "Fecha": fecha_str,
                "Mean_NDVI_Full": mean_ndvi_full,
                "Mean_NDVI_Masked": mean_ndvi_masked,
                "File": filename
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

df_res = pd.DataFrame(records)
print(f"Processed {len(df_res)} files.")
print("Sample of calculated NDVI:")
print(df_res.head(10))
print("\nMean of full:", df_res["Mean_NDVI_Full"].mean())
print("Mean of masked:", df_res["Mean_NDVI_Masked"].mean())
