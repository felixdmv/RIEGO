import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
path = os.path.join(base_path, "datos", "canduela", "RIEGOS_1.csv")

df = pd.read_csv(path)
print("Columns:", list(df.columns))

# Clean Agua m3
df['Agua m3'] = df['Agua m³'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['Agua m3'] = pd.to_numeric(df['Agua m3'], errors='coerce')

# Parse duration
def parse_duration(d):
    if pd.isna(d):
        return None
    # duration is in hh:mm:ss format
    try:
        parts = str(d).split(':')
        if len(parts) == 3:
            return float(parts[0]) + float(parts[1])/60.0 + float(parts[2])/3600.0
        elif len(parts) == 2:
            return float(parts[0]) + float(parts[1])/60.0
    except:
        return None
    return None

df['Duracion_Horas'] = df['Duración'].apply(parse_duration)

# Group by sector
print("\n--- SUMMARY BY SECTOR IN RIEGOS_1.csv ---")
for sector in sorted(df['Sectores de Riego'].dropna().unique()):
    sect_df = df[df['Sectores de Riego'] == sector].copy()
    print(f"\nSector: {sector}")
    print("  Records:", len(sect_df))
    print("  Total Water (m3):", sect_df['Agua m3'].sum())
    print("  Total Duration (hours):", sect_df['Duracion_Horas'].sum())
    
    # Calculate flow rate for non-zero records
    valid_df = sect_df[(sect_df['Agua m3'] > 0) & (sect_df['Duracion_Horas'] > 0)].copy()
    if len(valid_df) > 0:
        valid_df['Flow_m3_h'] = valid_df['Agua m3'] / valid_df['Duracion_Horas']
        print(f"  Calculated Flow Rate (m3/h) - Mean: {valid_df['Flow_m3_h'].mean():.2f}, Median: {valid_df['Flow_m3_h'].median():.2f}")
    else:
        print("  No valid records for flow rate calculation")
