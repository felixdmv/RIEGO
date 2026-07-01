import pandas as pd
df = pd.read_csv("datos/canduela/Dataset_Final_S2.csv")
print("Latest 5 rows in Dataset_Final_S2.csv:")
print(df.tail(5).to_string())
