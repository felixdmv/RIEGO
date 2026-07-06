import pandas as pd

p1 = 'datos/canduela/Datos estaciones Cesens 2026.xlsx'
p2 = 'datos/canduela/CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx'

df1 = pd.read_excel(p1)
df1['Parsed'] = pd.to_datetime(df1['Fecha'], errors='coerce')
df2 = pd.read_excel(p2)
df2['Parsed'] = pd.to_datetime(df2['Fecha'], errors='coerce')

print("Month counts in Datos estaciones Cesens 2026.xlsx:")
print(df1['Parsed'].dt.month.value_counts().sort_index())

print("\nMonth counts in CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx:")
print(df2['Parsed'].dt.month.value_counts().sort_index())

# Find if there is a gap or a missing period in df2
df2_sorted = df2.sort_values('Parsed')
df2_sorted['diff'] = df2_sorted['Parsed'].diff()
large_diffs = df2_sorted[df2_sorted['diff'] > pd.Timedelta(hours=2)]
if not large_diffs.empty:
    print("\nLarge gaps in CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx:")
    for idx, row in large_diffs.head(10).iterrows():
        prev_row = df2_sorted.loc[idx - 1] if idx - 1 in df2_sorted.index else None
        print(f"Gap from {prev_row['Fecha'] if prev_row is not None else 'N/A'} to {row['Fecha']} (Duration: {row['diff']})")
else:
    print("\nNo large gaps found (> 2 hours)")
