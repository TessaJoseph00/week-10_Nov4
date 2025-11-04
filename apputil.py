# apputil.py
import pickle
import pandas as pd
import numpy as np
import os

model1 = model2 = model3 = tfidf = None

# Load models only if the pickle files exist
if os.path.exists("model_1.pickle"):
    with open("model_1.pickle", "rb") as f:
        model1 = pickle.load(f)

if os.path.exists("model_2.pickle"):
    with open("model_2.pickle", "rb") as f:
        model2 = pickle.load(f)

if os.path.exists("model_3.pickle"):
    with open("model_3.pickle", "rb") as f:
        tfidf, model3 = pickle.load(f)


def roast_category(roast_value):
    """Map roast text to numerical category."""
    mapping = {
        "Very Light": 0,
        "Light": 1,
        "Medium-Light": 2,
        "Medium": 3,
        "Medium-Dark": 4,
        "Dark": 5,
        "Very Dark": 6,
    }
    return mapping.get(roast_value, np.nan)


def predict_rating(df_X, text=False):
    """Predict coffee rating using the appropriate model."""
    
    if text:
        if tfidf is None or model3 is None:
            raise ValueError("TF-IDF model not trained yet.")
        X_text = tfidf.transform(df_X["text"])
        return model3.predict(X_text)

    if model1 is None:
        raise ValueError("Linear Regression model not trained yet.")

    # If roast column exists, use model2 if available
    if "roast" in df_X.columns and model2 is not None:
        df_X = df_X.copy()
        df_X["roast_cat"] = df_X["roast"].map(roast_category)
        preds = np.zeros(len(df_X))

        # Rows with known roast
        mask_known = df_X["roast_cat"].notna()
        if mask_known.any():
            preds[mask_known] = model2.predict(
                df_X.loc[mask_known, ["100g_USD", "roast_cat"]]
            )

        # unknown roast
        mask_unknown = ~mask_known
        if mask_unknown.any():
            preds[mask_unknown] = model1.predict(df_X.loc[mask_unknown, ["100g_USD"]])
        return preds

    return model1.predict(df_X[["100g_USD"]])


if __name__ == "__main__":
    print("Numeric + Roast Example")
    df = pd.DataFrame(
        [
            [10.0, "Dark"],
            [15.0, "Very Light"],
            [12.5, "Unknown Roast"],
        ],
        columns=["100g_USD", "roast"],
    )
    try:
        print(predict_rating(df))
    except ValueError as e:
        print(e)

    print("\nText Example")
    df_text = pd.DataFrame(
        [
            ["A delightful coffee with hints of chocolate and caramel."],
            ["A strong coffee with a bold flavor and a smoky finish."],
        ],
        columns=["text"],
    )
    try:
        print(predict_rating(df_text, text=True))
    except ValueError as e:
        print(e)
