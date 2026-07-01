path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line_no, line in enumerate(lines, 1):
    if "model_dl" in line or "AgroTwoTower" in line or "modelo_percepcion" in line:
        print(f"Line {line_no}: {line.strip()}")
