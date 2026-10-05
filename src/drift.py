"""Reporte de deriva de datos (Evidently): entrenamiento (referencia) frente a prueba/producción."""
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.common import ROOT

ref = pd.read_csv(ROOT / "data" / "reference.csv")
cur = pd.read_csv(ROOT / "data" / (sys.argv[1] if len(sys.argv) > 1 else "current.csv"))
snap = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
snap.save_html(str(ROOT / "drift_report.html"))
m = snap.dict()["metrics"][0]["value"]
print(f"Columnas con deriva: {int(m['count'])} ({m['share']:.0%})")
