import rasterio
from rasterio.warp import transform
import glob
import os

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

# Extract lons and lats
lons = [c[0] for c in coords_s2]
lats = [c[1] for c in coords_s2]

# Project to UTM EPSG:32630
xs, ys = transform('EPSG:4326', 'EPSG:32630', lons, lats)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

print("Sector 2 UTM Bounds (EPSG:32630):")
print(f"  X range: {min_x:.1f} to {max_x:.1f}")
print(f"  Y range: {min_y:.1f} to {max_y:.1f}")

dir_old = r"datos/canduela/GEE_Tesis_ImageChips-20260217T075250Z-1-001/GEE_Tesis_ImageChips"
dir_new = r"datos/canduela/GEE_Tesis_NuevaParcela"

for name, directory in [("Old", dir_old), ("New", dir_new)]:
    files = glob.glob(os.path.join(directory, "*.tif"))
    if files:
        with rasterio.open(files[0]) as src:
            b = src.bounds
            print(f"\n{name} TIF Bounds:")
            print(f"  X range: {b.left:.1f} to {b.right:.1f}")
            print(f"  Y range: {b.bottom:.1f} to {b.top:.1f}")
            # Check overlap
            overlap_x = max(0, min(max_x, b.right) - max(min_x, b.left))
            overlap_y = max(0, min(max_y, b.top) - max(min_y, b.bottom))
            print(f"  Overlaps in X: {overlap_x > 0}, Overlaps in Y: {overlap_y > 0}")
