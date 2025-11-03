# apputil.py
import pickle
import pandas as pd
import numpy as np

with open("model_1.pickle", "rb") as f:
    model1 = pickle.load(f)

with open("model_2.pickle", "rb") as f:
    model2 = pickle.load(f)

with open("model_3.pickle", "rb") as f:
    tfidf, model3 = pickle.load(f)


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
    return mapping.get(roast_value, np.nan)


def predict_rating(df_X, text=False):
    if text:
        X_text = tfidf.transform(df_X["text"])
        return model3.predict(X_text)

    if "roast" in df_X.columns:
        df_X = df_X.copy()
        df_X["roast_cat"] = df_X["roast"].map(roast_category)
        preds = np.zeros(len(df_X))
        mask_known = df_X["roast_cat"].notna()
        if mask_known.any():
            preds[mask_known] = model2.predict(
                df_X.loc[mask_known, ["100g_USD", "roast_cat"]]
            )
        mask_unknown = ~mask_known
        if mask_unknown.any():
            preds[mask_unknown] = model1.predict(df_X.loc[mask_unknown, ["100g_USD"]])
        return preds

    return model1.predict(df_X[["100g_USD"]])


if __name__ == "__main__":
    print("=== Numeric + Roast Example ===")
    df = pd.DataFrame(
        [
            [10.0, "Dark"],
            [15.0, "Very Light"],
            [12.5, "Unknown Roast"],
        ],
        columns=["100g_USD", "roast"],
    )
    print(predict_rating(df))

    print("\n=== Text Example ===")
    df_text = pd.DataFrame(
        [
            ["A delightful coffee with hints of chocolate and caramel."],
            ["A strong coffee with a bold flavor and a smoky finish."],
        ],
        columns=["text"],
    )
    print(predict_rating(df_text, text=True))
