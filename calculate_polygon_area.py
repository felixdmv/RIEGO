import numpy as np
from shapely.geometry import Polygon
import pyproj
from shapely.ops import transform

# Coordinates of Sector 3
coords_s3 = [
    [41.81678686420481, -4.574917752384817],
    [41.818230196510825, -4.576967082310395],
    [41.81668046599235, -4.577445262519584],
    [41.81611098414579, -4.577826787646235],
    [41.81537961791841, -4.5758385605643825],
    [41.81678686420481, -4.574917752384817]
]

# Swap to (lon, lat) for shapely / pyproj
polygon_coords = [(c[1], c[0]) for c in coords_s3]
polygon = Polygon(polygon_coords)

# Project to UTM zone 30N (EPSG:32630) to get area in square meters
wgs84 = pyproj.CRS('EPSG:4326')
utm30n = pyproj.CRS('EPSG:32630')
project = pyproj.Transformer.from_crs(wgs84, utm30n, always_axis_order=True).transform
polygon_utm = transform(project, polygon)

print("Calculated UTM Area of Sector 3 (m2):", polygon_utm.area)
print("Expected Area of Sector 3 (m2):", 40650)
print("Ratio:", polygon_utm.area / 40650)
