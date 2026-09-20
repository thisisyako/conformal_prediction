import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop, comorbidity_cols):
        self.cols_to_drop = cols_to_drop
        self.comorbidity_cols = comorbidity_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        rcount_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5+': 5}
        X_out["rcount"] = X_out["rcount"].replace(rcount_mapping).astype(int)

        X_out["comorbidities"] = X_out[self.comorbidity_cols].sum(axis=1)
        X_out = X_out.drop(columns=self.cols_to_drop)
        return X_out

def build_prep_pipeline(comorbidities_vars=None):
    categorical_features = ['facid']

    numeric_features = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 
                        'bloodureanitro', 'creatinine', 'bmi', 'pulse', 
                        'respiration', 'comorbidities', 'rcount']

    comorbidities_vars = ["pneum", "asthma", "irondef", "hemo",
                    "malnutrition", "fibrosisandother", "depress",
                    "psychother", "psychologicaldisordermajor",
                    "substancedependence", "dialysisrenalendstage"] \
                        if comorbidities_vars is None else comorbidities_vars

    cols_to_drop = ["eid", "gender", "discharged", "vdate"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('engineering', FeatureEngineer(cols_to_drop=cols_to_drop, comorbidity_cols=comorbidities_vars)),
        ('preprocessing', preprocessor)
    ])

    return pipeline