import os

root_dir = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

for root, dirs, files in os.walk(root_dir):
    for f in files:
        if any(x in f.lower() for x in ["ndvi", "satelite", "radar", "lst", "sentinel", "modis"]):
            print(os.path.join(root, f))
