import os
import time
import pandas as pd

path = "datos/canduela/Dataset_Final_S2.csv"
if os.path.exists(path):
    print("File exists!")
    print("Modification time:", time.ctime(os.path.getmtime(path)))
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Max date:", df['Fecha'].max())
    # print last 3 rows
    print(df.tail(3)[['Fecha']])
else:
    print("File does not exist.")
