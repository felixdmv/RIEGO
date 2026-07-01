import pandas as pd

path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO\datos_multimodales_completos.csv"
try:
    df = pd.read_csv(path, nrows=5)
    print("Columnas de datos_multimodales_completos.csv:")
    print(list(df.columns))
    print("\nMuestra:")
    print(df)
    
    # Check if there is any column with plot id or site name
    df_all = pd.read_csv(path)
    if 'Parcela' in df_all.columns:
         print("\nValores únicos de Parcela:")
         print(df_all['Parcela'].unique())
    elif 'Sector' in df_all.columns:
         print("\nValores únicos de Sector:")
         print(df_all['Sector'].unique())
except Exception as e:
    print("Error reading:", e)
