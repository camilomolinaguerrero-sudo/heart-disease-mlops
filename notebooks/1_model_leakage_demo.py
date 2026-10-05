# %% [markdown]
# # Etapa 1. Preprocesamiento y detección de *data leakage*
# Dataset *Heart Failure Prediction* (918 pacientes, 11 variables, objetivo `HeartDisease`).
# Semilla 42, partición 80/20 estratificada.

# %%
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.common import *
raw = pd.read_csv(ROOT / "data" / "heart.csv")
print(raw.shape); raw.head()

# %% [markdown]
# ## 1. Exploración y valores faltantes

# %%
print(raw[TARGET].value_counts(normalize=True).round(3))
print("Colesterol = 0:", (raw.Cholesterol == 0).sum(), "| PA reposo = 0:", (raw.RestingBP == 0).sum())
raw.describe().T.round(2)

# %% [markdown]
# **Interpretación.** La clase positiva es el 55,3 %, así que el problema está casi balanceado y accuracy
# es una métrica razonable junto al AUC. Hay 172 pacientes con colesterol 0 y uno con presión 0: valores
# fisiológicamente imposibles que codifican datos ausentes. Se convierten en `NaN` y se imputan con la
# mediana **dentro del pipeline** (solo con datos de entrenamiento).

# %%
df = load_data()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="ChestPainType", hue=TARGET, ax=ax[0])
sns.countplot(data=df, x="ST_Slope", hue=TARGET, ax=ax[1])
sns.heatmap(df[NUM + [TARGET]].corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax[2])
plt.tight_layout(); plt.show()

# %% [markdown]
# **Interpretación.** La angina asintomática (`ASY`) y la pendiente plana/descendente del segmento ST se
# asocian con la enfermedad; entre las numéricas, `Oldpeak` correlaciona positivamente y `MaxHR`
# negativamente con el objetivo. Son las señales que los modelos deberían recuperar.

# %% [markdown]
# ## 2. Demostración de fuga de datos
# Se añade una variable artificial `leaky_feature = y + ruido`. Se compara (a) escalar con todo el
# conjunto antes de partir y dejar la fuga activa frente a (b) el flujo correcto con `Pipeline`.
# Para aislar los dos tipos de fuga se evalúan tres escenarios.

# %%
Xd = pd.get_dummies(df.drop(columns=TARGET), drop_first=False).astype(float)
Xd = Xd.fillna(Xd.median()); y = df[TARGET]
rng = np.random.default_rng(0)
leaky = y + rng.normal(0, 0.01, len(y))
grid_p = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}

def run_leaky(X):
    Xs = MinMaxScaler().fit_transform(X)                       # escalado con TODO el conjunto
    a, b, c, d = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=SEED)
    g = GridSearchCV(SVC(probability=True, random_state=SEED), grid_p, cv=5, scoring="roc_auc").fit(a, c)
    return roc_auc_score(d, g.predict_proba(b)[:, 1])

def run_clean(X):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    p = Pipeline([("sc", MinMaxScaler()), ("svc", SVC(probability=True, random_state=SEED))])
    g = GridSearchCV(p, {"svc__C": grid_p["C"], "svc__gamma": grid_p["gamma"]}, cv=5, scoring="roc_auc").fit(a, c)
    return roc_auc_score(d, g.predict_proba(b)[:, 1])

X_leak = Xd.assign(leaky_feature=leaky)
res = pd.Series({
    "Con variable fuga + escalado global": run_leaky(X_leak),
    "Con variable fuga + Pipeline": run_clean(X_leak),
    "Sin variable fuga + escalado global": run_leaky(Xd),
    "Sin variable fuga + Pipeline (correcto)": run_clean(Xd)}).round(4)
res

# %% [markdown]
# **Interpretación.** La variable artificial lleva el AUC prácticamente a 1, y eso ocurre aunque se use
# `Pipeline`: la fuga por una variable que contiene la respuesta no la corrige ninguna técnica de
# validación, solo la revisión de las variables (una AUC casi perfecta en datos clínicos reales es señal
# de alarma). El escalado global, en cambio, es una fuga sutil: la diferencia entre las filas
# "escalado global" y "Pipeline" sin variable fuga es pequeña, porque MinMax filtra solo mínimos y
# máximos, pero el procedimiento correcto es el único que garantiza que la estimación de prueba sea
# honesta.

# %% [markdown]
# ## 3. Comparación de modelos sin fuga (Pipeline + GridSearchCV)

# %%
X_train, X_test, y_train, y_test = split(df)
modelos = {
    "SVC": (SVC(probability=True, random_state=SEED), {"clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.1, "scale"]}),
    "LogisticRegression": (LogisticRegression(max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "RandomForest": (RandomForestClassifier(random_state=SEED), {"clf__n_estimators": [200, 500], "clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5]}),
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]}),
}
filas, ajustados = [], {}
for nombre, (m, pg) in modelos.items():
    g = train_pipeline(X_train, y_train, m, pg)
    p = g.predict_proba(X_test)[:, 1]
    ajustados[nombre] = g
    filas.append({"modelo": nombre, "AUC_cv": g.best_score_, "AUC_test": roc_auc_score(y_test, p),
                  "Accuracy_test": accuracy_score(y_test, p > .5), "mejores_params": g.best_params_})
ranking = pd.DataFrame(filas).sort_values("AUC_test", ascending=False).reset_index(drop=True)
ranking.index += 1
ranking.round(4)

# %% [markdown]
# **Interpretación.** El ranking se ordena por AUC en prueba, pero la AUC de validación cruzada
# (`AUC_cv`) es la que se usó para elegir hiperparámetros y es la más estable: con 184 pacientes de
# prueba, una diferencia de ~0,01 en AUC está dentro del ruido muestral, de modo que los modelos
# vecinos en el ranking no deben considerarse distintos sin una prueba pareada (DeLong). Aquí KNN lidera en
# prueba (0,935) pero RandomForest lidera en validación cruzada (0,932): las cinco AUC de prueba caen en un
# rango de 0,02, de modo que ningún modelo se impone con claridad. SVC queda último en ambos criterios.

# %%
plt.figure(figsize=(7, 3.5))
ranking.set_index("modelo")[["AUC_cv", "AUC_test"]].plot.barh(ax=plt.gca()); plt.xlim(.8, 1)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
