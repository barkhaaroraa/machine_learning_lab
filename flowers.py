import pandas as pd
import matplotlib as mp
import random

file=pd.read_csv("iris.csv")

file.rename(columns={"species":"labels"}, inplace=True)
file.drop(columns="Id",inplace=True)
# print(file.head())
# print(file.info())
# test_indices=file.index.tolist()

# print(random.sample(population=indices,k=20))
indices=file.index.tolist()
test_size=20
test_indices=random.sample(population=indices,k=test_size)

test=file.loc[20]
# print(test)

train=file.drop(test_indices)
# print(train)

if isinstance(test_size,float):
    test_size=round(test_size*len(file))

def train_test_split(file, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(file))
        print(test_size)
    indices=file.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_file = file.loc[test_indices]
    train_file = file.drop(test_indices)
    print("yes")
    return train_file, test_file

train_file,test_file=train_test_split(file,test_size=0.1)

