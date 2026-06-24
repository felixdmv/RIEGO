import os

img_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Imágenes"
print("Scanning:", img_dir)
if os.path.exists(img_dir):
    try:
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                path = os.path.join(root, f)
                print(os.path.relpath(path, img_dir), f"({os.path.getsize(path)} bytes)")
    except Exception as e:
        print("Error walking Images:", e)
else:
    print("Images folder does not exist")
