import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
excel_path = os.path.join(base_path, "datos", "canduela", "CANDUELA - 080221084930-077_ (1).xlsx")

df = pd.read_excel(excel_path)

# Assign a unique event ID to each row based on when Equipo is not null
df['event_id'] = df['Equipo'].notna().cumsum()

# Group by event_id and find unique valves
event_valves = df.groupby('event_id')['Válvulas'].apply(lambda x: list(x.dropna().unique()))

multi_valve_events = event_valves[event_valves.apply(len) > 1]
print("Number of events with multiple valves:", len(multi_valve_events))
if len(multi_valve_events) > 0:
    print("Sample multi-valve events:")
    print(multi_valve_events.head(10))
else:
    print("Each event only irrigates one valve!")
    
# Let's compute the flow rate by associating the header's Consumo real and Duración directly with the single valve
event_headers = df[df['Equipo'].notna()].copy()
# Find the single valve for each event
df['valvula_filled'] = df['Válvulas'].ffill() # Wait, if we group by event_id, we can get the valve
event_valves_single = df.groupby('event_id')['Válvulas'].first() # First non-null valve in the event
event_headers['Valvula'] = event_headers['event_id'].map(event_valves_single)

print("\nEvent headers columns:", list(event_headers.columns))

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
event_headers['Duracion_Horas'] = event_headers[duracion_col].apply(parse_duration)

event_headers_valid = event_headers[(event_headers[consumo_real_col] > 0) & (event_headers['Duracion_Horas'] > 0)].copy()
event_headers_valid['Caudal_m3_h'] = event_headers_valid[consumo_real_col] / event_headers_valid['Duracion_Horas']

for valve in event_headers_valid['Valvula'].dropna().unique():
    v_df = event_headers_valid[event_headers_valid['Valvula'] == valve]
    print(f"\n--- VALVE: {valve} ---")
    print("Events count:", len(v_df))
    print(f"  Calculated Flow Rate (m3/h) - Mean: {v_df['Caudal_m3_h'].mean():.3f}, Median: {v_df['Caudal_m3_h'].median():.3f}")
    print(f"  Min: {v_df['Caudal_m3_h'].min():.3f}, Max: {v_df['Caudal_m3_h'].max():.3f}")
