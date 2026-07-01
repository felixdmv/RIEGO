import pandas as pd
import os

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"
df = pd.read_csv(os.path.join(base_path, 'datos', 'Dataset_Final_S3.csv'), parse_dates=['Fecha'])

# We want to find rolling 30-day windows with high standard deviation and unique values in FP_Potencial_Min
df = df.sort_values('Fecha').reset_index(drop=True)
df['Std_30d'] = df['FP_Potencial_Min'].rolling(30, center=True).std()
df['Unique_30d'] = df['FP_Potencial_Min'].rolling(30, center=True).apply(lambda x: len(x.unique()))

# Find top 10 dates with high std and at least 15 unique values in the 30-day window
varying_df = df[(df['Std_30d'] > 0.1) & (df['Unique_30d'] > 15)].sort_values('Std_30d', ascending=False)

print("Varying periods found:", len(varying_df))
if not varying_df.empty:
    print("\nTop 15 dates with high local variation in water potential:")
    print(varying_df[['Fecha', 'FP_Potencial_Min', 'Std_30d', 'Unique_30d']].head(15))
else:
    print("\nNo periods with unique > 15 and std > 0.1 found. Let's list the top 20 rows with lowest (most negative) potentials:")
    print(df.sort_values('FP_Potencial_Min')[['Fecha', 'FP_Potencial_Min']].head(20))
