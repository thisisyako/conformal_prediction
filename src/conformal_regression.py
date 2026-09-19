from sklearn.linear_model import QuantileRegressor

def train_quantile_regressor(X, y, quantile=0.5):
    model = QuantileRegressor(quantile=quantile)

    print("Training QuantileRegressor...")
    model.fit(X, y)
    print("QuantileRegressor has been trained.")

    return model
