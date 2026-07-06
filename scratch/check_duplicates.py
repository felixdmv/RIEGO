import pandas as pd

p1 = 'datos/canduela/Datos estaciones Cesens 2026.xlsx'
p2 = 'datos/canduela/CANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx'

df1 = pd.read_excel(p1)
df2 = pd.read_excel(p2)

print("Datos estaciones Cesens 2026.xlsx:")
print("  Total rows:", len(df1))
print("  Unique Fechas:", df1['Fecha'].nunique())
print("  Duplicate Fechas count:", df1['Fecha'].duplicated().sum())

print("\nCANDUELA MINI 1 ALMENDRO 2026 ENERO-JULIO.xlsx:")
print("  Total rows:", len(df2))
print("  Unique Fechas:", df2['Fecha'].nunique())
print("  Duplicate Fechas count:", df2['Fecha'].duplicated().sum())

if df1['Fecha'].duplicated().sum() > 0:
    print("\nDuplicate sample in Datos estaciones Cesens 2026.xlsx:")
    print(df1[df1['Fecha'].duplicated(keep=False)].head(6))
