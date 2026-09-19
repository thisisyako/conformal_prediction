import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing import build_prep_pipeline
from src.conformal_regression import train_quantile_regressor #, calibrate_quantile_regressor, predict_quantile_regressor
# from src.evaluation import compute_coverage, compute_average_width

def main():
    df = pd.read_csv("data/raw/LengthOfStay.csv")

    X_raw = df.drop("lengthofstay", axis=1)
    target =  df["lengthofstay"]
    X_train_raw, X_temp, y_train, y_temp = train_test_split(X_raw, target, train_size=0.8, random_state=42)
    X_calibrate_raw, X_test_raw, y_calibrate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Created X_train (size={len(X_train_raw)}), X_calibrate (size={len(X_calibrate_raw)}), X_test (size={len(X_test_raw)})")
    pipeline = build_prep_pipeline()
    X_train = pipeline.fit_transform(X_train_raw)
    X_calibrate = pipeline.transform(X_calibrate_raw)
    X_test = pipeline.transform(X_test_raw)


if __name__ == "__main__" :
    main() 