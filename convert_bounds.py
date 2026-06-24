import pyproj

# EPSG:32630 to EPSG:4326
utm = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
# Bounds: BoundingBox(left=368950.0, bottom=4630470.0, right=369200.0, top=4630800.0)

lon1, lat1 = utm(368950.0, 4630470.0, inverse=True)
lon2, lat2 = utm(369200.0, 4630800.0, inverse=True)

print("Bounding Box in WGS84:")
print(f"Bottom-Left: Lat {lat1}, Lon {lon1}")
print(f"Top-Right: Lat {lat2}, Lon {lon2}")

# Let's check where the user's coordinate falls relative to these bounds:
user_lat = 41.814040723556154
user_lon = -4.577624452593563

print("\nUser's Coordinate:")
print(f"Lat: {user_lat}, Lon: {user_lon}")
print(f"Is within Lon bounds? {lon1 <= user_lon <= lon2}")
print(f"Is within Lat bounds? {lat1 <= user_lat <= lat2}")

# Convert user coords to UTM
user_x, user_y = utm(user_lon, user_lat)
print(f"\nUser UTM: X={user_x}, Y={user_y}")
print(f"X within bounds? {368950.0 <= user_x <= 369200.0}")
print(f"Y within bounds? {4630470.0 <= user_y <= 4630800.0}")
