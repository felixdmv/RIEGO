import re
import base64
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos"
html_path = os.path.join(base_path, "RIEGO - copia", "mapa_tesis_final.html")
out_png_path = os.path.join(base_path, "RIEGO", "map_overlay.png")

print("HTML Path:", html_path)
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

match = re.search(r'data:image/png;base64,([a-zA-Z0-9+/=]+)', content)
if match:
    b64_data = match.group(1)
    print("Found base64 length:", len(b64_data))
    # Decode and save
    img_data = base64.b64decode(b64_data)
    with open(out_png_path, "wb") as out_f:
        out_f.write(img_data)
    print("Decoded and saved to:", out_png_path)
else:
    print("Base64 image overlay pattern not found in HTML.")
