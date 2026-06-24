import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)

# Forward fill header columns
header_cols = [df.columns[i] for i in [0, 2, 4, 5, 6, 9, 10]]
for col in header_cols:
    df[col] = df[col].ffill()

# Filter by valve V2
valvulas_col = df.columns[27]
df_v2 = df[df[valvulas_col] == 'V2'].copy()

# Parse duration to hours
def parse_duration(d):
    if pd.isna(d):
        return None
    try:
        parts = str(d).split(':')
        if len(parts) == 3:
            return float(parts[0]) + float(parts[1])/60.0 + float(parts[2])/3600.0
        elif len(parts) == 2:
            return float(parts[0]) + float(parts[1])/60.0
    except:
        return None
    return None

duracion_col = df.columns[6]
consumo_real_col = df.columns[10]
df_v2['Duracion_Horas'] = df_v2[duracion_col].apply(parse_duration)

# Sort by Comienzo if available
comienzo_col = df.columns[4]
df_v2 = df_v2.sort_values(by=comienzo_col)

print("--- RECORDS FOR V2 ---")
print("Total rows:", len(df_v2))
print("Sample rows:")
print(df_v2[[comienzo_col, duracion_col, 'Duracion_Horas', consumo_real_col]].head(30))
