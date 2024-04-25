import pandas as pd
import numpy as np

df = pd.read_csv('exp3.csv')

print("Original Data:")
print(df)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Price'].fillna(df['Price'].mean(), inplace=True)

df['Color'].fillna(df['Color'].mode()[0], inplace=True)
df['Size'].fillna(df['Size'].mode()[0], inplace=True)

average_price = df[df['Price'] != 4500]['Price'].mean()
df['Price'] = df['Price'].replace(4500, average_price)

print("\nPreprocessed Data:")
print(df)