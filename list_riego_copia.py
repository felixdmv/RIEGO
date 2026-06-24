import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO - copia"
print("Scanning:", base_path)
if os.path.exists(base_path):
    try:
        for item in os.listdir(base_path):
            path = os.path.join(base_path, item)
            if os.path.isdir(path):
                print(f"[Dir]  {item}")
            else:
                print(f"[File] {item} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error listing RIEGO - copia:", e)
else:
    print("RIEGO - copia does not exist")
