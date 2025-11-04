# train.py
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


def roast_category(roast_value):
    mapping = {
        "Very Light": 0,
        "Light": 1,
        "Medium-Light": 2,
        "Medium": 3,
        "Medium-Dark": 4,
        "Dark": 5,
        "Very Dark": 6,
    }
    return mapping.get(roast_value, None)


def train_models():
    url = (
        "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    )
    df = pd.read_csv(url)
    y = df["rating"].values

    # Linear Regression
    X1 = df[["100g_USD"]]

    lr = LinearRegression()
    lr.fit(X1, y)

    with open("model_1.pickle", "wb") as f:
        pickle.dump(lr, f)
    print(" model_1.pickle saved.")

    #  Decision Tree Regressor
    df["roast_cat"] = df["roast"].map(roast_category)
    X2 = df[["100g_USD", "roast_cat"]]

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X2, y)

    with open("model_2.pickle", "wb") as f:
        pickle.dump(dtr, f)
    print(" model_2.pickle saved.")

    # TF-IDF + Linear Regression
    df["desc_3"] = df["desc_3"].fillna("")
    tfidf = TfidfVectorizer(max_features=500)
    X_text = tfidf.fit_transform(df["desc_3"])
    
    lr_text = LinearRegression()
    lr_text.fit(X_text, y)

    with open("model_3.pickle", "wb") as f:
        pickle.dump((tfidf, lr_text), f)
    print(" model_3.pickle saved.")


if __name__ == "__main__":
    train_models()
