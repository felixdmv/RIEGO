from PIL import Image
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
img_path = os.path.join(base_path, "map_overlay.png")

print("Image path:", img_path)
print("File exists:", os.path.exists(img_path))

try:
    with Image.open(img_path) as img:
        print("Format:", img.format)
        print("Size:", img.size)
        print("Mode:", img.mode)
except Exception as e:
    print("Error opening image:", e)
