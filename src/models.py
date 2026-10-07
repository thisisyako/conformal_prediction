import joblib
import os

from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def get_or_train_qr(path_lo, path_hi, X_train, y_train, alpha):
    # On récupère les modèles de QR si ils sont déjà entraînés, sinon on les entraîne.
    if os.path.exists(path_lo) and os.path.exists(path_hi):
        with open(path_lo, "rb+") as f:
            QR_lo = joblib.load(f)
        with open(path_hi, "rb+") as f:
            QR_hi = joblib.load(f)
    else:
        QR_lo, QR_hi = train_quantile_regressors(X_train, y_train, alpha, path_lo, path_hi)

    return QR_lo, QR_hi

def train_quantile_regressors(X_train, y_train, alpha, path_lo, path_hi):
    """
    This function trains two quantile regressors at their respectively lower and upper bound
    set by LOWER_QUANTILE and UPPER_QUANTILE.
    """
    
    QR_model_lo = train_quantile_regressor(X_train, y_train, alpha)
    print("Trained QR_lo model")

    QR_model_hi = train_quantile_regressor(X_train, y_train, 1 - alpha)
    print("Trained QR_hi model")

    save_model(QR_model_lo, path_lo)
    save_model(QR_model_hi, path_hi)

    return QR_model_lo, QR_model_hi


def train_quantile_regressor(X, y, quantile=0.5):
    model = QuantileRegressor(quantile=quantile)

    print("Training QuantileRegressor...")
    model.fit(X, y)
    print("QuantileRegressor has been trained.")
    return model


def train_hgbr(X, y):
    model = HistGradientBoostingRegressor(loss="poisson")

    print("Training HGB Regressor...")
    model.fit(X, y)
    print("HGB Regressor has been trained.")
    return model

def save_model(model, path):
    with open(path, "wb") as f:
        joblib.dump(model, f, protocol=5)
