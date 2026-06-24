import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
gee_dir = os.path.join(base_path, "datos", "canduela", "GEE_Tesis_ImageChips-20260217T075250Z-1-001")

print("--- NON-TIF FILES IN GEE DIR ---")
for root, dirs, files in os.walk(gee_dir):
    for f in files:
        if not f.endswith(".tif"):
            path = os.path.join(root, f)
            print(os.path.relpath(path, base_path), f"({os.path.getsize(path)} bytes)")
