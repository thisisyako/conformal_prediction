import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing import build_prep_pipeline

def create_data():
    df = pd.read_csv("data/raw/LengthOfStay.csv")
    
    X_raw = df.drop("lengthofstay", axis=1)
    target =  df["lengthofstay"]

    X_train_raw, X_temp, y_train, y_temp = train_test_split(X_raw, target, train_size=0.8, random_state=42)
    X_calibrate_raw, X_test_raw, y_calibrate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Created X_train (size={len(X_train_raw)}), X_calibrate (size={len(X_calibrate_raw)}), X_test (size={len(X_test_raw)})")

    # This part is for the Grouped Conformal Prediction, allows to check each group's interval
    gender_calibrate = X_calibrate_raw["gender"].reset_index(drop=True)
    gender_test = X_test_raw["gender"].reset_index(drop=True)

    pipeline_train = build_prep_pipeline(cols_to_drop=["gender"])
    
    X_train = pipeline_train.fit_transform(X_train_raw)
    X_calibrate = pipeline_train.transform(X_calibrate_raw)
    X_test = pipeline_train.transform(X_test_raw)

    print(f"Data has been transformed with the pipeline")

    return X_train, X_calibrate, X_test, y_train, y_calibrate, y_test, gender_calibrate, gender_test