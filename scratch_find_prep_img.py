path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\run_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "def _prepare_img" in line:
            print(f"Line {idx}: {line.strip()}")
            break
