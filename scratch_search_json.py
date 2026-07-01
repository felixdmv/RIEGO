import json
import os

json_files = [f for f in os.listdir(".") if f.endswith(".json")]

for jf in json_files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert to string to search easily
            data_str = json.dumps(data, ensure_ascii=False).lower()
            if "carralancha" in data_str or "v4" in data_str or "pozaldez" in data_str:
                print(f"=== Found in {jf} ===")
                # If it's a dict or list, search specific parts
                if isinstance(data, list):
                    for idx, item in enumerate(data[:5]):
                        print(f"Item {idx}: {str(item)[:300]}")
                elif isinstance(data, dict):
                    for k, v in list(data.items())[:5]:
                        print(f"Key '{k}': {str(v)[:300]}")
    except Exception as e:
        print(f"Error reading {jf}: {e}")
