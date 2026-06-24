import math

coords_s3 = [
    [41.81678686420481, -4.574917752384817],
    [41.818230196510825, -4.576967082310395],
    [41.81668046599235, -4.577445262519584],
    [41.81611098414579, -4.577826787646235],
    [41.81537961791841, -4.5758385605643825],
    [41.81678686420481, -4.574917752384817]
]

# Flat approximation
lat_avg = sum(c[0] for c in coords_s3) / len(coords_s3)
lat_to_m = 111132.0
lon_to_m = 111132.0 * math.cos(math.radians(lat_avg))

# Convert coords to meters relative to the first point
x = []
y = []
for c in coords_s3:
    y.append(c[0] * lat_to_m)
    x.append(c[1] * lon_to_m)

# Shoelace formula for polygon area
def shoelace_area(x, y):
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0

calc_area = shoelace_area(x, y)
print("Flat Calculated Area of Sector 3 (m2):", calc_area)
print("Expected Area of Sector 3 (m2):", 40650)
print("Ratio:", calc_area / 40650)
print("Area in ha:", calc_area / 10000.0)
