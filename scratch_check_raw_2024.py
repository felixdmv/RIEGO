import pandas as pd
import glob
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
datos_path = os.path.join(base_path, 'datos')

excel_files = glob.glob(os.path.join(datos_path, 'florapulse', "*2024*.xlsx"))
print("Found 2024 files:", [os.path.basename(f) for f in excel_files])

for f in excel_files:
    try:
        xl = pd.ExcelFile(f)
        df = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df.columns if 'fecha' in str(c).lower()][0]
        pot_col = [c for c in df.columns if 'potencial' in str(c).lower() and 'almendra' in str(c).lower() and 'baseline' not in c.lower()][0]
        
        # Keep only date and potential
        df = df[[fecha_col, pot_col]].dropna()
        df[pot_col] = pd.to_numeric(df[pot_col], errors='coerce')
        df = df.dropna()
        
        print(f"\n--- File: {os.path.basename(f)} ---")
        print("Total non-null rows:", len(df))
        if len(df) > 0:
            print("Summary stats for potential:")
            print(df[pot_col].describe())
            print("Negative potential counts (potential < 0.49):", len(df[df[pot_col] < 0.49]))
            # Print sample of negative potentials
            if len(df[df[pot_col] < 0.49]) > 0:
                print("Sample negative rows:")
                print(df[df[pot_col] < 0.49].head(5))
    except Exception as e:
        print(f"Error reading {os.path.basename(f)}: {e}")
