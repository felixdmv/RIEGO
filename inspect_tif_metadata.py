import rasterio
import os
import glob

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
gee_dir = os.path.join(base_path, "datos", "canduela", "GEE_Tesis_ImageChips-20260217T075250Z-1-001", "GEE_Tesis_ImageChips")

tif_files = glob.glob(os.path.join(gee_dir, "*.tif"))
print("Total TIF files:", len(tif_files))
if len(tif_files) > 0:
    first_tif = tif_files[0]
    print("Inspecting first TIF:", os.path.basename(first_tif))
    try:
        with rasterio.open(first_tif) as src:
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)
            print("Transform:", src.transform)
            print("Width, Height:", src.width, src.height)
            print("Count of bands:", src.count)
    except Exception as e:
        print("Error reading TIF:", e)
