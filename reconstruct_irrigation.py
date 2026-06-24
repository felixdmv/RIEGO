import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)

# Use column indices directly
equipo_col = df.columns[0]
programa_col = df.columns[2]
comienzo_col = df.columns[4]
finalizacion_col = df.columns[5]
duracion_col = df.columns[6]
consumo_prog_col = df.columns[9]
consumo_real_col = df.columns[10]
valvulas_col = df.columns[27]

print(f"Equipo col: {repr(equipo_col)}")
print(f"Programa col: {repr(programa_col)}")
print(f"Comienzo col: {repr(comienzo_col)}")
print(f"Finalización col: {repr(finalizacion_col)}")
print(f"Duración col: {repr(duracion_col)}")
print(f"Consumo prog col: {repr(consumo_prog_col)}")
print(f"Consumo real col: {repr(consumo_real_col)}")
print(f"Valvulas col: {repr(valvulas_col)}")

header_cols = [equipo_col, programa_col, comienzo_col, finalizacion_col, duracion_col, consumo_prog_col, consumo_real_col]

for col in header_cols:
    df[col] = df[col].ffill()

# Drop rows where Válvulas is null (these were the headers themselves)
df_clean = df.dropna(subset=[valvulas_col]).copy()

print("Reconstructed shape:", df_clean.shape)
print("Unique valves after reconstruction:", df_clean[valvulas_col].unique())

# Parse duration to hours
def parse_duration(d):
    if pd.isna(d):
        return None
    if isinstance(d, str):
        parts = d.split(':')
        if len(parts) == 3:
            try:
                return float(parts[0]) + float(parts[1])/60.0 + float(parts[2])/3600.0
            except:
                return None
        elif len(parts) == 2:
            try:
                return float(parts[0]) + float(parts[1])/60.0
            except:
                return None
    elif hasattr(d, 'hour'):
        return d.hour + d.minute/60.0 + d.second/3600.0
    return None

df_clean['Duracion_Horas'] = df_clean[duracion_col].apply(parse_duration)

# Filter records where Consumo real (m3) and Duracion_Horas are positive
df_valid = df_clean[(df_clean[consumo_real_col] > 0) & (df_clean['Duracion_Horas'] > 0)].copy()
df_valid['Caudal_m3_h'] = df_valid[consumo_real_col] / df_valid['Duracion_Horas']

for valve in df_clean[valvulas_col].unique():
    v_df = df_valid[df_valid[valvulas_col] == valve]
    print(f"\n--- VALVE: {valve} ---")
    print("Valid rows count:", len(v_df))
    if len(v_df) > 0:
        print("  Duración (horas) - Mean:", v_df['Duracion_Horas'].mean(), "Median:", v_df['Duracion_Horas'].median())
        print("  Consumo real (m3) - Mean:", v_df[consumo_real_col].mean(), "Median:", v_df[consumo_real_col].median())
        print("  Calculated Flow Rate (m3/h) - Mean:", v_df['Caudal_m3_h'].mean(), "Median:", v_df['Caudal_m3_h'].median())
        print("  Min:", v_df['Caudal_m3_h'].min(), "Max:", v_df['Caudal_m3_h'].max())
