import os
import glob

print("Archivos .env:")
print(glob.glob(".env*"))

print("\nVariables de entorno relevantes:")
for k, v in os.environ.items():
    if any(x in k.upper() for x in ["AEMET", "API", "KEY", "TOKEN", "EARTH", "EE"]):
        print(f"{k}: {v[:10]}...{v[-5:]}" if len(v) > 15 else f"{k}: {v}")
