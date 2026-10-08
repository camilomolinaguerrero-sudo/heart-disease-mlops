# Heart Disease MLOps

Proyecto integrador de aprendizaje automático (sección 10.10 del libro del curso): predicción de falla
cardíaca (`HeartDisease` = 1/0) con el conjunto *Heart Failure Prediction* (918 pacientes, 11 variables),
desde el análisis de fuga de datos hasta la API, el contenedor, Kubernetes, la integración continua y el
monitoreo de deriva. Semilla 42; partición 80/20 estratificada (734 pacientes de entrenamiento, 184 de prueba).

Código: <https://github.com/camilomolinaguerrero-sudo/heart-disease-mlops>

| Etapa | Capítulo | Resultado |
|---|---|---|
| 0 | Estructura del repositorio | `app/`, `src/`, `docker/`, `k8s/`, `tests/`, `notebooks/` |
| 1 | [Fuga de datos](01_fuga_de_datos) | Una variable que contiene la respuesta lleva el AUC a 1,000 aunque se use `Pipeline` |
| 2 | [Modelado con validación segura](02_modelado_cv) | RandomForest, AUC de validación cruzada 0,9315, AUC de prueba 0,928 |
| 3–5 | [API, Docker, Kubernetes y CI](03_despliegue) | Servicio FastAPI en contenedor, desplegado en kind; CI en verde |
| 6 | [Monitoreo de deriva](04_monitoreo) | 0 de 11 columnas con deriva en la partición real; 3 de 11 con datos alterados |

Datos: `heart.csv` proviene de un espejo público en GitHub (Kaggle exige inicio de sesión); dos espejos
independientes dieron el mismo SHA-1. Hay 172 pacientes con colesterol = 0 y uno con presión en reposo = 0:
son valores imposibles que codifican datos ausentes, así que se convierten en `NaN` y se imputan con la
mediana dentro del `Pipeline`, solo con los datos de entrenamiento.
