import pandas as pd

p1 = 'datos/canduela/Datos estaciones Cesens 2026.xlsx'
p2 = 'datos/canduela/CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx'

xl1 = pd.ExcelFile(p1)
df1 = xl1.parse(xl1.sheet_names[0])
df1['Parsed'] = pd.to_datetime(df1['Fecha'], errors='coerce')

xl2 = pd.ExcelFile(p2)
df2 = xl2.parse(xl2.sheet_names[0])
df2['Parsed'] = pd.to_datetime(df2['Fecha'], errors='coerce')

print("=== FILE 1: Datos estaciones Cesens 2026.xlsx ===")
print("Date range:", df1['Parsed'].min(), "to", df1['Parsed'].max())
print("Rows:", len(df1))
print("Diffs between consecutive datetimes (value counts):")
print(df1['Parsed'].diff().value_counts().head(5))

print("\n=== FILE 2: CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx ===")
print("Date range:", df2['Parsed'].min(), "to", df2['Parsed'].max())
print("Rows:", len(df2))
print("Diffs between consecutive datetimes (value counts):")
print(df2['Parsed'].diff().value_counts().head(5))
