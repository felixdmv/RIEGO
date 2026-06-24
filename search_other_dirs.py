import os

paths_to_scan = [
    r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Escritorio",
    r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Imágenes",
    r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO - copia"
]

print("--- SCANNING OTHER DIRECTORIES ---")
for p in paths_to_scan:
    print(f"\nScanning: {p}")
    if os.path.exists(p):
        try:
            for root, dirs, files in os.walk(p):
                for f in files:
                    low_f = f.lower()
                    if any(k in low_f for k in ["mapa", "riego", "canduela", "sector", "valvula", "v2"]):
                        path = os.path.join(root, f)
                        print(f"Match: {os.path.relpath(path, p)} ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"Error walking {p}: {e}")
    else:
        print("Path does not exist")
