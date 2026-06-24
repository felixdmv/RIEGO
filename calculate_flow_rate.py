import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)
print("Columns:", list(df.columns))

# Clean duration: standard format is hh:mm:ss or similar. Let's inspect Duración values.
print("\nSample 'Duración' values:")
print(df['Duración'].dropna().head(10))

# Let's inspect rows for each valve
for valve in ['Valvula1 (V1)', 'V2', 'v3 (V3)', 'v4 (V4)']:
    df_valve = df[df['Válvulas'] == valve].copy()
    print(f"\n--- VALVE: {valve} ---")
    print("Number of records:", len(df_valve))
    print("Non-null Consumo real (m3):", df_valve['Consumo real (m3)'].notna().sum())
    print("Non-null Duración:", df_valve['Duración'].notna().sum())
    
    # Let's parse duration to hours
    def parse_duration(d):
        if pd.isna(d):
            return None
        # Could be time object or string
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
        elif hasattr(d, 'hour'): # datetime.time object
            return d.hour + d.minute/60.0 + d.second/3600.0
        return None

    df_valve['Duracion_Horas'] = df_valve['Duración'].apply(parse_duration)
    # Filter rows where Duracion_Horas > 0 and Consumo real > 0
    df_valid = df_valve[(df_valve['Duracion_Horas'] > 0) & (df_valve['Consumo real (m3)'] > 0)].copy()
    if len(df_valid) > 0:
        df_valid['Caudal_m3_h'] = df_valid['Consumo real (m3)'] / df_valid['Duracion_Horas']
        print(f"Calculated flow rates (m3/h) for {valve}:")
        print("  Mean:", df_valid['Caudal_m3_h'].mean())
        print("  Median:", df_valid['Caudal_m3_h'].median())
        print("  Min:", df_valid['Caudal_m3_h'].min())
        print("  Max:", df_valid['Caudal_m3_h'].max())
        print("  Sample rows:")
        print(df_valid[['Duración', 'Duracion_Horas', 'Consumo real (m3)', 'Caudal_m3_h']].head(5))
    else:
        print("No valid rows for flow rate calculation")
