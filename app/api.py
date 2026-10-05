from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load(Path(__file__).parent / "model.joblib")
app = FastAPI(title="Heart Failure Risk API")


class Patient(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: float = Field(gt=0)
    Cholesterol: float = Field(ge=0, description="0 se trata como dato ausente")
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(gt=0)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Patient):
    X = pd.DataFrame([data.model_dump()])
    X.loc[X["Cholesterol"] == 0, "Cholesterol"] = float("nan")
    proba = float(model.predict_proba(X)[0][1])
    return {"heart_disease_probability": proba, "prediction": int(proba > 0.5)}
