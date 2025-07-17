import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Inline display
%matplotlib inline

# Load datasets
ratings = pd.read_csv("../app/data/ratings.csv")
items = pd.read_csv("../app/data/items.csv")

print("Ratings shape:", ratings.shape)
print("Items shape:", items.shape)
