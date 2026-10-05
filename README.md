# Heart Disease MLOps (local)

Proyecto integrador de aprendizaje automático: predicción de falla cardíaca (`HeartDisease` = 1/0) con el
dataset *Heart Failure Prediction* (918 pacientes, 11 variables), desde el análisis exploratorio hasta
API, contenedor, Kubernetes, CI y monitoreo de deriva. Semilla 42; partición 80/20 estratificada
(734 entrenamiento, 184 prueba).

## Estructura

```
app/api.py                   API FastAPI (/predict, /health) + model.joblib
docker/                      Dockerfile y requirements.txt (versiones exactas)
k8s/                         deployment.yaml, service.yaml
notebooks/                   1_model_leakage_demo, 2_model_pipeline_cv (.py fuente + .ipynb ejecutado)
src/                         common.py (carga, pipeline, GridSearch), drift.py, simular_deriva.py, py2nb.py
tests/                       pruebas de la API y de la carga/partición
.github/workflows/ci.yml     lint + pruebas + docker build
data/                        heart.csv, reference.csv (train), current.csv (test)
drift_report.html            deriva entrenamiento vs prueba
drift_report_simulated.html  deriva con datos alterados a propósito
model.joblib                 pipeline final
```

## Datos

`heart.csv` proviene de un espejo público en GitHub (Kaggle exige inicio de sesión); dos espejos
independientes dieron el mismo SHA-1. Hay 172 pacientes con colesterol = 0 y uno con presión en reposo
= 0: son valores imposibles que codifican datos ausentes, así que se convierten en `NaN` y se imputan con
la mediana dentro del `Pipeline` (solo con entrenamiento). Preprocesamiento: mediana + MinMax en las
numéricas, one-hot (`handle_unknown="ignore"`) en las categóricas.

## Etapa 1: fuga de datos

AUC de SVC en prueba (cuaderno 1):

| Escenario | AUC |
|---|---|
| Variable artificial `y + ruido` + escalado global | 1,000 |
| Variable artificial + Pipeline | 1,000 |
| Sin variable artificial + escalado global | 0,928 |
| Sin variable artificial + Pipeline (correcto) | 0,928 |

La variable que contiene la respuesta lleva el AUC a 1 aunque se use `Pipeline`: ninguna técnica de
validación corrige una variable contaminada, solo la revisión de las variables. El escalado global es una
fuga real pero aquí casi inocua (MinMax solo filtra mínimos y máximos).

Ranking sin fuga (Pipeline + GridSearchCV, cv=5 estratificado, AUC ROC):

| Modelo | AUC cv | AUC prueba | Accuracy prueba |
|---|---|---|---|
| KNN | 0,913 | 0,935 | 0,886 |
| LogisticRegression | 0,923 | 0,931 | 0,897 |
| GradientBoosting | 0,927 | 0,929 | 0,897 |
| RandomForest | 0,932 | 0,928 | 0,880 |
| SVC | 0,919 | 0,914 | 0,832 |

Las cinco AUC de prueba difieren en 0,02 con solo 184 pacientes, por lo que ningún modelo se impone
con claridad; SVC queda último en ambos criterios.

## Etapa 2: modelo seleccionado

Entre LogisticRegression, RandomForest y GradientBoosting se eligió **RandomForest** (`max_depth=5`,
`min_samples_leaf=3`, `n_estimators=200`) solo por AUC de validación cruzada (0,9315). En prueba:
AUC 0,928, accuracy 0,880, precision 0,870, recall 0,922, F1 0,895. AUC de entrenamiento 0,961. La prueba
de permutación (100 permutaciones) da AUC 0,500 ± 0,029 con la respuesta barajada frente a 0,9315
real (p = 0,0099, el mínimo posible con 100 permutaciones).

## Etapa 3: API y Docker

La API recibe campos con nombre y validados (el pipeline necesita las columnas originales; la lista
numérica del enunciado no sirve con variables categóricas). Colesterol = 0 se trata como ausente.

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"Age":54,"Sex":"M","ChestPainType":"ASY","RestingBP":140,"Cholesterol":239,"FastingBS":0,"RestingECG":"Normal","MaxHR":120,"ExerciseAngina":"Y","Oldpeak":1.5,"ST_Slope":"Flat"}'
# {"heart_disease_probability":0.947,"prediction":1}
```

Verificado: la imagen se construye y el contenedor responde a `/health` y `/predict`. Se usa
`python:3.11-slim` (el enunciado propone 3.10) porque scikit-learn 1.9 lo exige; las versiones
de `requirements.txt` coinciden con las que entrenaron `model.joblib`.

## Etapa 4: Kubernetes

```bash
kind create cluster --name heart
kind load docker-image heart-api:latest --name heart
kubectl apply -f k8s/
kubectl port-forward svc/heart-service 8080:80
```

Verificado con kind (Docker Desktop): el pod queda `1/1 Running`, la sonda de disponibilidad pasa y,
vía `port-forward`, `/health` devuelve `{"status":"ok"}` y `/predict` devuelve probabilidad 0,947 y
predicción 1, igual que el contenedor suelto. El servicio `LoadBalancer` queda con EXTERNAL-IP
`<pending>` porque kind no tiene proveedor de balanceador; se accede por `port-forward`.

## Etapa 5: CI

`.github/workflows/ci.yml`: flake8, pytest (6 pruebas, pasan localmente) y `docker build`. No se
ha ejecutado en GitHub: el repositorio aún no se ha subido.

## Etapa 6: monitoreo (Evidently 0.7)

```bash
python -m src.drift current.csv      # genera drift_report.html
python -m src.drift drifted.csv      # tras: python -m src.simular_deriva
```

Entrenamiento frente a prueba: 0 de 11 columnas con deriva, como corresponde a una partición aleatoria.
Con datos alterados a propósito (edad +12, presión ×1,15, frecuencia máxima ×0,85): 3 de 11 columnas
(27 %) con deriva; el umbral de alerta por defecto es 50 %, así que el reporte detecta las tres
columnas pero no dispara alerta global.

## Reproducir

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r docker/requirements.txt pandas matplotlib seaborn pytest httpx flake8 evidently jupytext nbclient ipykernel
python -m ipykernel install --user --name heart-mlops
python src/py2nb.py notebooks/1_model_leakage_demo.py notebooks/2_model_pipeline_cv.py
pytest tests/ && flake8 app src tests
```
