# Etapas 3, 4 y 5: API, Docker, Kubernetes y CI

## API y contenedor

La API (`app/api.py`, FastAPI) recibe campos con nombre y validados, porque el pipeline necesita las
columnas originales; una lista numérica no sirve con variables categóricas. Un colesterol igual a 0 se trata
como ausente, igual que en el entrenamiento.

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"Age":54,"Sex":"M","ChestPainType":"ASY","RestingBP":140,"Cholesterol":239,"FastingBS":0,"RestingECG":"Normal","MaxHR":120,"ExerciseAngina":"Y","Oldpeak":1.5,"ST_Slope":"Flat"}'
# {"heart_disease_probability":0.947,"prediction":1}
```

La imagen usa `python:3.11-slim` (el enunciado propone 3.10) porque scikit-learn 1.9 lo exige; las versiones
de `docker/requirements.txt` son las mismas que entrenaron `model.joblib`.

## Kubernetes (kind)

```bash
kind create cluster --name heart
kind load docker-image heart-api:latest --name heart
kubectl apply -f k8s/
kubectl port-forward svc/heart-service 8080:80
```

Verificado: el pod queda `1/1 Running`, la sonda de disponibilidad pasa y, por `port-forward`, `/health`
devuelve `{"status":"ok"}` y `/predict` devuelve probabilidad 0,947 y predicción 1, igual que el contenedor
suelto. El servicio `LoadBalancer` queda con `EXTERNAL-IP <pending>` porque kind no trae proveedor de
balanceador, por lo que se accede por `port-forward`.

## Integración continua

`.github/workflows/ci.yml` ejecuta flake8, las 6 pruebas de pytest y `docker build` en cada *push*. La
primera ejecución sobre `main` terminó con éxito.
