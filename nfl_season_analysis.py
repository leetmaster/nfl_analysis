import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
nfl = pd.read_csv('season_2021.csv')

# Codificar resultados
result_encoder = {'result': {'W': 1, 'T': 0, 'L': 0}}

# encode result column using encoder
nfl.replace(result_encoder, inplace=True)
nfl.infer_objects(copy=False)

# check result value counts
nfl['result'].value_counts()

# change stat to view plot
stat = 'TotYd_defense'

# box plot of stat
stat_plot = sns.boxplot(hue='result', x='result', y=stat, data=nfl, legend=False)

# plot labels
stat_plot.set_xticks([0,1], ['loss/tie', 'win'])

plt.show()
# list feature names
print(nfl.columns[8:])

# Seleccionar features y estandarizar
features = nfl.iloc[:, 8:]  # Ajusta según tu dataset
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = nfl['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Crear y entrenar modelo XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Para clasificación binaria
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión XGBoost: {accuracy*100:.1f}%')

# Grid de parámetros para XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# Búsqueda de mejores parámetros
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f'Mejores parámetros: {grid_search.best_params_}')
print(f'Mejor precisión CV: {grid_search.best_score_*100:.1f}%')

# Modelo optimizado
opt_xgb = grid_search.best_estimator_

test_sizes = [val/100 for val in range(20, 36)]
best_acc = 0
best_size = 0.25

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Usar los mejores parámetros encontrados
    model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f'Precisión: {acc*100:.1f}% | test_size = {test_size}')

    if acc > best_acc:
        best_acc = acc
        best_size = test_size

print(f'\nMejor test_size: {best_size} con precisión: {best_acc*100:.1f}%')

# Importancia por ganancia (más informativa que los coeficientes de LR)
importance = opt_xgb.feature_importances_
feature_names = features.columns

# Visualizar
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names, hue=feature_names, legend=False)
plt.title('Importancia de Características - XGBoost')
plt.xlabel('Importancia (Gain)')
plt.ylabel('Estadística')
plt.show()

# Tabla resumen
for i, (name, imp) in enumerate(zip(feature_names, importance)):
    print(f'Feature: {name}, Importancia: {imp:.3f}')

# Cargar datos nuevos
new_data = pd.read_csv('seahawks_2025.csv')
#new_data.fillna(0, inplace=True)

# Estandarizar con el mismo scaler
new_X = new_data.loc[:, features.columns]
new_X_sc = scaler.transform(new_X)

# Predecir con XGBoost optimizado
new_preds = opt_xgb.predict(new_X_sc)
new_results = new_data['result'].astype(float)

# Evaluar
acc_new = accuracy_score(new_results, new_preds)
print(f'Precisión en datos 2025: {acc_new*100:.1f}%')

# select only game data
col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
game_data = new_data.loc[:,col_names]
# create comparison table
comp_table = game_data.assign(predicted = new_preds,
                              actual = new_results.astype(int))

# set team abbreviation (in capitals) and year
team = 'Seattle Seahawks'
year = 2025

# print title and table
print(f'Predicted Wins vs Actual Wins for {team} in {year}')
comp_table
