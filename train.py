# train.py
"""
Train and save two models:
Linear Regression model (price → rating)
Decision Tree Regressor (price + roast → rating)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# data loading
URL = "https://raw.githubusercontent.com/MuseumofModernData/coffee-quality-database/main/coffee_ratings.csv"
df = pd.read_csv(URL)

# Features and target
X1 = df[["100g_USD"]].values
y = df["rating"].values

# Train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X1, y)

# Save model
with open("model_1.pickle", "wb") as file:
    pickle.dump(linear_reg, file)

print("model_1.pickle saved.")

# Decision Tree Regressor

def roast_category(roast):
    """Convert roast text into numeric labels."""
    if roast == "Light":
        return 0
    if roast == "Medium-Light":
        return 1
    if roast == "Medium":
        return 2
    if roast == "Medium-Dark":
        return 3
    if roast == "Dark":
        return 4
    return np.nan  


# Convert roast text to numeric
df["roast_cat"] = df["roast"].apply(roast_category)

# Features and target
X2 = df[["100g_USD", "roast_cat"]].values
y2 = df["rating"].values

# Train Decision Tree
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X2, y2)

# Save model
with open("model_2.pickle", "wb") as file:
    pickle.dump(decision_tree, file)

print("model_2.pickle saved.")
