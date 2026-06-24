import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO - copia\datos"
print("Scanning:", base_path)
if os.path.exists(base_path):
    try:
        for root, dirs, files in os.walk(base_path):
            for f in files:
                path = os.path.join(root, f)
                print(os.path.relpath(path, base_path), f"({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error walking copia/datos:", e)
else:
    print("copia/datos does not exist")
