import pandas as pd
import os
import glob

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# 1. Dataset_Final_S3.csv
path_df = os.path.join(base_path, "datos", "canduela", "Dataset_Final_S3.csv")
if os.path.exists(path_df):
    df = pd.read_csv(path_df)
    print("Dataset_Final_S3.csv date range:")
    print("  Min:", df['Fecha'].min())
    print("  Max:", df['Fecha'].max())
    print("  Columns:", list(df.columns))

# 2. Florapulse Excel files date range
fp_dir = os.path.join(base_path, "datos", "canduela", "florapulse")
excel_files = glob.glob(os.path.join(fp_dir, "*.xlsx"))
print("\nFlorapulse files date ranges:")
for f in sorted(excel_files):
    try:
        xl = pd.ExcelFile(f)
        df_fp = xl.parse(xl.sheet_names[0])
        fecha_col = [c for c in df_fp.columns if 'fecha' in str(c).lower()][0]
        df_fp['Parsed_Fecha'] = pd.to_datetime(df_fp[fecha_col], errors='coerce')
        print(f"  File {os.path.basename(f)}:")
        print(f"    Min: {df_fp['Parsed_Fecha'].min()}")
        print(f"    Max: {df_fp['Parsed_Fecha'].max()}")
    except Exception as e:
        print(f"  Error reading {os.path.basename(f)}: {e}")

# 3. New Excel file date range
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")
if os.path.exists(excel_path):
    df_xl = pd.read_excel(excel_path)
    comienzo_col = df_xl.columns[4]
    df_xl['Parsed_Comienzo'] = pd.to_datetime(df_xl[comienzo_col], errors='coerce')
    print("\nNew Excel file date range:")
    print("  Min:", df_xl['Parsed_Comienzo'].min())
    print("  Max:", df_xl['Parsed_Comienzo'].max())
