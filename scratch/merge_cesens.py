import pandas as pd
import os

p1 = 'datos/canduela/Datos estaciones Cesens 2026.xlsx'
p2 = r'C:\Users\felix\Downloads\Datos estaciones Cesens (1).xlsx'
output_p = 'datos/canduela/Datos estaciones Cesens 2026.xlsx'

# Read both files
xl1 = pd.ExcelFile(p1)
df1 = xl1.parse(xl1.sheet_names[0])
df1['Parsed'] = pd.to_datetime(df1['Fecha'], errors='coerce')

xl2 = pd.ExcelFile(p2)
df2 = xl2.parse(xl2.sheet_names[0])
df2['Parsed'] = pd.to_datetime(df2['Fecha'], errors='coerce')

print(f"Base file rows: {len(df1)}, Min: {df1['Parsed'].min()}, Max: {df1['Parsed'].max()}")
print(f"New file rows: {len(df2)}, Min: {df2['Parsed'].min()}, Max: {df2['Parsed'].max()}")

# Concatenate
df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged = df_merged.sort_values('Parsed')

# Drop duplicates keeping the first occurrence (since df1 is complete, it will be preferred,
# and for dates only in df2 it will use df2)
df_merged = df_merged.drop_duplicates(subset=['Fecha'], keep='first')
df_merged = df_merged.drop(columns=['Parsed'])

print(f"Merged file rows: {len(df_merged)}")

# Double check range
df_merged['Parsed'] = pd.to_datetime(df_merged['Fecha'], errors='coerce')
print(f"Merged date range: {df_merged['Parsed'].min()} to {df_merged['Parsed'].max()}")

# Look for any gaps greater than 2 hours in merged data
df_merged_sorted = df_merged.sort_values('Parsed')
df_merged_sorted['diff'] = df_merged_sorted['Parsed'].diff()
large_diffs = df_merged_sorted[df_merged_sorted['diff'] > pd.Timedelta(hours=2)]
if not large_diffs.empty:
    print("Warning: Large gaps found in merged data:")
    for idx, row in large_diffs.iterrows():
        prev_row = df_merged_sorted.loc[idx - 1] if idx - 1 in df_merged_sorted.index else None
        print(f"  Gap from {prev_row['Fecha'] if prev_row is not None else 'N/A'} to {row['Fecha']} ({row['diff']})")
else:
    print("No gaps greater than 2 hours found in merged data.")

df_merged = df_merged.drop(columns=['Parsed'])

# Save to excel
with pd.ExcelWriter(output_p) as writer:
    df_merged.to_excel(writer, sheet_name=xl1.sheet_names[0], index=False)
print(f"Saved merged Excel to {output_p}")
