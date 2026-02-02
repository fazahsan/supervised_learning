# Crear modelos ML y compararlos con escala de características
#%%
# Importar librerías
from data import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#librerias para evaluar los modelos
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# %%
# Preparar los datos
X = df.drop('EVENTO_MUERTE', axis=1)
y = df['EVENTO_MUERTE']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# # Escalar las características
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# %%
# Definir los modelos a comparar , son algoritmos clasicos de ML
# Define pipelines
pipelines = {
    'Regresión logística': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ]),
    
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ]),
    
    'Árbol de decisión': Pipeline([
        ('model', DecisionTreeClassifier())
    ]),
    
    'Bosque aleatorio': Pipeline([
        ('model', RandomForestClassifier())
    ]),
    
    'Aumento de gradiente': Pipeline([
        ('model', GradientBoostingClassifier())
    ]),
    
    'Bayes ingenuo': Pipeline([
        ('scaler', StandardScaler()),  # optional but safe
        ('model', GaussianNB())
    ])
}

# %%
# Entrenar y evaluar los modelos
results = {}
for name, model in pipelines.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} - Precisión: {accuracy:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)
# %%
# Con cross-validation
cv_results = {}
cv_res={}
for name, model in pipelines.items():
    cv_score = cross_val_score(model, X, y, cv=5)
    cv_results[name] = cv_score.mean()
    cv_res[name] = cv_score
    print(f"{name} - Precisión CV: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
# %%
# visualizar los resultados
results_df = pd.DataFrame({name: scores for name, scores in cv_res.items()})
results_df.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)
plt.ylabel('Precisión')
plt.title('Comparación de modelos de ML con Validación Cruzada')
plt.tight_layout()
plt.show()
# %%
# Ajuste de hiperparámetros con GridSearchCV para el mejor modelo (por ejemplo, SVM)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])

param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X, y)

print("Mejor precisión:", grid.best_score_)
print("Mejores parámetros:", grid.best_params_)

# %%
