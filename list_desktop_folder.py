import os

desktop_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Escritorio"
print("Scanning:", desktop_dir)
if os.path.exists(desktop_dir):
    try:
        for root, dirs, files in os.walk(desktop_dir):
            for f in files:
                path = os.path.join(root, f)
                print(os.path.relpath(path, desktop_dir), f"({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error walking Desktop:", e)
else:
    print("Desktop folder does not exist")
