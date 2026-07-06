import os
import pandas as pd
import glob

def check_xlsx(p):
    try:
        xl = pd.ExcelFile(p)
        print(f"File: {os.path.basename(p)}")
        print(f"  Sheets: {xl.sheet_names}")
        df = xl.parse(xl.sheet_names[0])
        fcol = [c for c in df.columns if 'fecha' in str(c).lower() or 'comienzo' in str(c).lower() or 'inicio' in str(c).lower()]
        if fcol:
            df['Parsed'] = pd.to_datetime(df[fcol[0]], errors='coerce')
            print(f"  Date range: {df['Parsed'].min()} to {df['Parsed'].max()}, rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
        else:
            print(f"  No date column. Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Error reading {os.path.basename(p)}: {e}")

print("=== CHECKING DOWNLOADED CESENS EXCEL ===")
cesens_download = r"C:\Users\felix\Downloads\Datos estaciones Cesens (1).xlsx"
if os.path.exists(cesens_download):
    check_xlsx(cesens_download)
else:
    print(f"{cesens_download} does not exist")

print("\n=== SEARCHING FOR CSV/TXT/ZIP IN DOWNLOADS ===")
downloads_dir = r"C:\Users\felix\Downloads"
for root, dirs, files in os.walk(downloads_dir):
    for f in files:
        if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.zip'):
            p = os.path.join(root, f)
            print(f"Found: {os.path.relpath(p, downloads_dir)} ({os.path.getsize(p)} bytes)")
