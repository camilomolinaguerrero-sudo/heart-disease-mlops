# Etapa 6: monitoreo de deriva

Se compara la distribución de entrenamiento (referencia) con la de datos nuevos usando Evidently 0.7.

```bash
python -m src.drift current.csv      # genera drift_report.html
python -m src.drift drifted.csv      # tras: python -m src.simular_deriva
```

| Comparación | Columnas con deriva |
|---|---|
| Entrenamiento frente a prueba | 0 de 11 |
| Datos alterados a propósito (edad +12, presión ×1,15, frecuencia máxima ×0,85) | 3 de 11 (27 %) |

Que la partición aleatoria no muestre deriva es lo esperado: ambas muestras salen de la misma población. Con
los datos alterados el reporte señala exactamente las tres columnas modificadas, pero no dispara alerta
global, porque el umbral por defecto exige que derive el 50 % de las columnas. En producción convendría
alertar por columna o bajar ese umbral.

Reportes completos: [entrenamiento frente a prueba](drift_report.html) ·
[datos alterados](drift_report_simulated.html)
