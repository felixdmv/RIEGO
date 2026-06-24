import os

onedrive_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos"
print("Scanning OneDrive folder for images and PDFs:", onedrive_dir)

exts = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".geojson", ".kml", ".shp"]
matches = []

checked = 0
for root, dirs, files in os.walk(onedrive_dir):
    if any(p in root for p in [".git", ".vscode", "node_modules", "anaconda3", "AppData", "Mendeley", "Grabaciones"]):
        continue
    for f in files:
        checked += 1
        low_f = f.lower()
        if any(low_f.endswith(e) for e in exts):
            path = os.path.join(root, f)
            # Skip Sentinel-2 chips to avoid massive print
            if "GEE_Tesis" in path:
                continue
            matches.append(path)

print(f"Checked {checked} files. Found {len(matches)} matching files (excluding GEE chips):")
for m in matches:
    print(f"  {os.path.relpath(m, onedrive_dir)} ({os.path.getsize(m)} bytes)")
