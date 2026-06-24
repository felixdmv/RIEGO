import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)

# Assign event ID
df['event_id'] = df['Equipo'].notna().cumsum()

# Find unique valves per event
event_valves = df.groupby('event_id')['Válvulas'].apply(lambda x: list(x.dropna().unique()))

# Keep only events with exactly 1 valve
single_valve_events = event_valves[event_valves.apply(len) == 1]
print("Events with exactly 1 valve:", len(single_valve_events))

# Map single valve to event headers
event_headers = df[df['Equipo'].notna()].copy()
event_headers['Valvula'] = event_headers['event_id'].map(single_valve_events.apply(lambda x: x[0]))

# Drop events with multiple or zero valves
event_headers = event_headers.dropna(subset=['Valvula'])

# Parse duration
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
event_headers['Duracion_Horas'] = event_headers[duracion_col].apply(parse_duration)

# Calculate flow rate
event_headers['Caudal_m3_h'] = event_headers[consumo_real_col] / event_headers['Duracion_Horas']

# Filter reasonable flow rates (e.g. between 1 and 200 m3/h)
for valve in sorted(event_headers['Valvula'].unique()):
    v_df = event_headers[event_headers['Valvula'] == valve].copy()
    print(f"\n--- VALVE: {valve} ---")
    print("  Total single-valve events:", len(v_df))
    # Filter flow rates > 0 and not null
    v_df_valid = v_df[(v_df['Caudal_m3_h'] > 0) & (v_df['Caudal_m3_h'].notna())]
    print("  Valid events:", len(v_df_valid))
    print(f"  Percentiles (m3/h):")
    print(f"    10th: {v_df_valid['Caudal_m3_h'].quantile(0.10):.3f}")
    print(f"    25th: {v_df_valid['Caudal_m3_h'].quantile(0.25):.3f}")
    print(f"    50th (Median): {v_df_valid['Caudal_m3_h'].quantile(0.50):.3f}")
    print(f"    75th: {v_df_valid['Caudal_m3_h'].quantile(0.75):.3f}")
    print(f"    90th: {v_df_valid['Caudal_m3_h'].quantile(0.90):.3f}")
    
    # Filter standard flow rates (around median) to find the most common value
    median_val = v_df_valid['Caudal_m3_h'].median()
    close_df = v_df_valid[(v_df_valid['Caudal_m3_h'] >= median_val * 0.8) & (v_df_valid['Caudal_m3_h'] <= median_val * 1.2)]
    print(f"  Events within 20% of median: {len(close_df)}")
    print(f"  Mean of close events: {close_df['Caudal_m3_h'].mean():.3f}")
    print("  Sample consumptions (m3):", close_df[consumo_real_col].head(5).tolist())
    print("  Sample durations (hours):", close_df['Duracion_Horas'].head(5).tolist())
