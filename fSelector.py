
import numpy as np
import pandas as pd
from collections import Counter


print('Medicina')
path = 'datasets/projects/medicina_18_24.csv'
df = pd.read_csv(path)
print(df.shape)
df = df.loc[df["vigencia_months"]>=23]
print(df.shape)
labels = list(df['label'])
print(Counter(labels))
print('\n\n')

print('Odontologia')
path = 'datasets/projects/odontologia_18_24.csv'
df = pd.read_csv(path)
print(df.shape)
df = df.loc[df["vigencia_months"]>=23]
print(df.shape)
labels = list(df['label'])
print(Counter(labels))
print('\n\n')

print('Veterinaria')
path = 'datasets/projects/veterinaria_18_24.csv'
df = pd.read_csv(path)
print(df.shape)
df = df.loc[df["vigencia_months"]>=23]
print(df.shape)
labels = list(df['label'])
print(Counter(labels))
print('\n\n')