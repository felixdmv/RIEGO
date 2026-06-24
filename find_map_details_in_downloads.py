import os
import sys
import re

# Safely print unicode
sys.stdout.reconfigure(encoding='utf-8')

downloads_dir = r"C:\Users\fdemiguel\Downloads"
keywords = ["canduela", "sector", "valvula", "v2", "caudal", "superficie", "area", "riego", "41.814", "-4.577"]

print("--- SEARCHING DOWNLOADS FOR MAP AND IRRIGATION ---")
for root, dirs, files in os.walk(downloads_dir):
    # Skip directories with too many irrelevant files
    if any(p in root for p in ["node_modules", "anaconda3", "AppData", "Mendeley"]):
        continue
    for f in files:
        low_f = f.lower()
        if any(k in low_f for k in keywords):
            path = os.path.join(root, f)
            try:
                size = os.path.getsize(path)
                print(f"File: {os.path.relpath(path, downloads_dir)} ({size} bytes)")
                
                # If Excel, inspect sheets
                if f.endswith((".xlsx", ".xls")):
                    import pandas as pd
                    try:
                        xl = pd.ExcelFile(path)
                        print("  Sheets:", xl.sheet_names)
                    except:
                        pass
            except Exception as e:
                print(f"  Error reading file: {e}")
