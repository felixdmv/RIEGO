with open("descargar_aemet_sardon.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("First 30 lines:")
for i, l in enumerate(lines[:30]):
    print(f"{i+1}: {l.strip()}")
