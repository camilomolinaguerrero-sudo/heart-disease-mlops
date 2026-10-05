"""Utilidades compartidas: carga, partición, preprocesamiento y entrenamiento."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TARGET = "HeartDisease"
NUM = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CAT = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def load_data(path=None):
    """Lee heart.csv. Colesterol = 0 y PA en reposo = 0 son valores imposibles: se marcan NaN."""
    df = pd.read_csv(path or ROOT / "data" / "heart.csv")
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = np.nan
    df.loc[df["RestingBP"] == 0, "RestingBP"] = np.nan
    return df


def split(df, test_size=0.2):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def train_pipeline(X_train, y_train, model, param_grid, cv=5):
    """Pipeline(preprocesamiento + modelo) optimizado con GridSearchCV por AUC ROC."""
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", model)])
    skf = StratifiedKFold(cv, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid
