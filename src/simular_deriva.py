"""Genera data/drifted.csv: pacientes de mayor edad, más presión y menos frecuencia máxima."""
import pandas as pd

from src.common import ROOT

d = pd.read_csv(ROOT / "data" / "current.csv")
d["Age"] += 12
d["RestingBP"] *= 1.15
d["MaxHR"] *= 0.85
d.to_csv(ROOT / "data" / "drifted.csv", index=False)
