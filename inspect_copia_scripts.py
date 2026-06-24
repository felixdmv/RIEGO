import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO - copia"
keywords = ["sector", "v2", "valvula", "superficie", "caudal", "m2", "ha", "litro", "metro", "flow", "area"]

print("--- SEARCH IN COPIA SCRIPTS ---")
for f in os.listdir(base_path):
    if f.endswith(".py"):
        path = os.path.join(base_path, f)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
            for kw in keywords:
                if kw.lower() in content.lower():
                    print(f"File: {f}, Keyword: {kw}")
                    lines = content.split('\n')
                    for idx, line in enumerate(lines):
                        if kw.lower() in line.lower() and any(x in line.lower() for x in ["2", "v2", "valvula", "caudal", "superficie", "area"]):
                            print(f"  Line {idx+1}: {line.strip()[:120]}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
