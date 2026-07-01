import math

coordenadas = [
  [-4.811755842895575, 41.37409504652195],
  [-4.812917239399024, 41.37351537008459],
  [-4.811444593553215, 41.37225123589853],
  [-4.810167967851344, 41.37273024205082],
  [-4.811755842895575, 41.37409504652195]
]

def polygon_area_m2(coords):
    # Convert lat/lon to meters using a local projection or spherical approximation
    # Since it's a small area, we can project to UTM zone 30N (or simple flat earth projection)
    # Latitude: 41.373, Longitude: -4.811
    # 1 degree lat = 111132 m
    # 1 degree lon = 111132 * cos(lat) m
    lat_avg = math.radians(41.373)
    r_lat = 111132.0
    r_lon = 111132.0 * math.cos(lat_avg)
    
    # Project to meters relative to the first coordinate
    x = []
    y = []
    for lon, lat in coords:
        x.append(lon * r_lon)
        y.append(lat * r_lat)
        
    # Shoelace formula
    n = len(coords)
    area = 0.0
    for i in range(n - 1):
        area += x[i] * y[i+1] - x[i+1] * y[i]
    return abs(area) / 2.0

area = polygon_area_m2(coordenadas)
print(f"Área calculada de la finca Carralancha: {area:.2f} m² ({area/10000:.4f} hectáreas)")
