import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
# Mateen Ulhaq & Lee
df = pd.read_csv('data/Iris.csv', sep = ',', header = None)

# print(df.head())

# https://stackoverflow.com/questions/48641632/extracting-specific-columns-from-pandas-dataframe
# kepy97
cols = [1,2,3,4]

training_rows = df[df.columns[cols]]


print(training_rows)