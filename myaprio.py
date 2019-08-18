
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    a=[]
    for j in range(0, 20):
       a.append(str(dataset.values[i,j]))
    transactions.append(a)

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
result = list(rules)
for rule in result:
    for item in rule:
        print(item)
        
                
