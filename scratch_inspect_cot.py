import json

path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\dataset_v4_cot.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 3:
            break
        print(f"\n--- Linea {idx} ---")
        try:
            data = json.loads(line)
            # Print a structured snippet
            print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
        except Exception as e:
            print("Error parsing JSON:", e)
            print(line[:500])
