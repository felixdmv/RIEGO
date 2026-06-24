import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
canduela_dir = os.path.join(base_path, "datos", "canduela")

print("--- CANDUELA FILES (EXCLUDING CHIPS) ---")
for root, dirs, files in os.walk(canduela_dir):
    if "GEE_Tesis" in root:
        continue
    for f in files:
        path = os.path.join(root, f)
        print(os.path.relpath(path, base_path), f"({os.path.getsize(path)} bytes)")
