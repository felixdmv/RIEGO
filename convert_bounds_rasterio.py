import rasterio
from rasterio.warp import transform
import os
import glob

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
gee_dir = os.path.join(base_path, "datos", "canduela", "GEE_Tesis_ImageChips-20260217T075250Z-1-001", "GEE_Tesis_ImageChips")

tif_files = glob.glob(os.path.join(gee_dir, "*.tif"))
if len(tif_files) > 0:
    first_tif = tif_files[0]
    with rasterio.open(first_tif) as src:
        bounds = src.bounds
        crs = src.crs
        
        # Transform bounds from UTM EPSG:32630 to WGS84 EPSG:4326
        # transform takes xs, ys
        xs = [bounds.left, bounds.right]
        ys = [bounds.bottom, bounds.top]
        
        lons, lats = transform(crs, 'EPSG:4326', xs, ys)
        print("Bounding Box in WGS84:")
        print(f"Lons: {lons}")
        print(f"Lats: {lats}")
        
        # Check user coord
        user_lat = 41.814040723556154
        user_lon = -4.577624452593563
        
        # Transform user coord to UTM EPSG:32630
        user_xs, user_ys = transform('EPSG:4326', crs, [user_lon], [user_lat])
        print("\nUser Coordinate in UTM:")
        print(f"X: {user_xs[0]}, Y: {user_ys[0]}")
        print(f"X within bounds? {bounds.left <= user_xs[0] <= bounds.right}")
        print(f"Y within bounds? {bounds.bottom <= user_ys[0] <= bounds.top}")
