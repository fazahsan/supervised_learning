# Predicción de Mortalidad por Insuficiencia Cardíaca usando Machine Learning

Este proyecto aplica **algoritmos clásicos de Machine Learning** para la **predicción de mortalidad** en pacientes con insuficiencia cardíaca, utilizando el dataset *Heart Failure Clinical Records*.  
El objetivo principal es **comparar distintos modelos**, evaluar su rendimiento mediante **validación cruzada** y analizar las **limitaciones inherentes al conjunto de datos**.

---

## Dataset

- **Nombre:** Heart Failure Clinical Records Dataset  
- **Número de muestras:** 299  
- **Número de características:** 12  
- **Variable objetivo:** `EVENTO_MUERTE` (clasificación binaria)
  - 0 → No fallecimiento
  - 1 → Fallecimiento

El dataset contiene variables clínicas y de laboratorio recogidas de pacientes con insuficiencia cardíaca.

---

## Algoritmos utilizados

Se evaluaron los siguientes modelos de Machine Learning:

- Regresión Logística
- Support Vector Machine (SVM) con kernel RBF
- K-Nearest Neighbors (KNN)
- Árbol de Decisión
- Bosque Aleatorio (Random Forest)
- Naive Bayes (Bayes ingenuo)
- Gradient Boosting (Aumento de gradiente)

---

## Metodología

1. **Separación de datos**
   - División entrenamiento / prueba (80 % / 20 %)

2. **Preprocesamiento**
   - Normalización mediante **Z-score (StandardScaler)** para los modelos sensibles a la escala
   - Uso de **Pipelines** para evitar *data leakage*

3. **Evaluación**
   - Validación cruzada con **5 folds**
   - Métricas utilizadas:
     - Accuracy (precisión)
     - Precision, Recall y F1-score
     - Matriz de confusión

4. **Optimización**
   - Ajuste de hiperparámetros para SVM (`C` y `gamma`) mediante GridSearchCV

---

## Resultados principales (Validación Cruzada)

| Modelo | Precisión CV |
|------|-------------|
| Regresión Logística | ~0.78 |
| SVM (optimizado) | ~0.79 |
| Bayes ingenuo | ~0.75 |
| Bosque Aleatorio | ~0.70 |
| Árbol de Decisión | ~0.64 |
| KNN | ~0.62 |
| Gradient Boosting | ~0.62 |

---

## Análisis de resultados

- La **regresión logística** obtuvo el mejor equilibrio entre precisión, estabilidad e interpretabilidad.
- El **SVM** mostró alta estabilidad y mejoró tras el ajuste de hiperparámetros.
- Las **matrices de confusión** revelan una **presencia consistente de falsos negativos** en todos los modelos.
- Este comportamiento sugiere un **solapamiento significativo entre clases**, lo que impone un **límite superior al rendimiento** alcanzable.

> La precisión se mantiene en el rango **70–78 %**, sin alcanzar valores del 90–95 %, debido principalmente al **tamaño reducido del dataset**, el **desbalance moderado de clases** y el **solapamiento clínico entre pacientes**.

---

## Conclusiones

- El rendimiento de los modelos está limitado por la **naturaleza del dataset**, no por la implementación.
- Modelos más complejos no garantizan mejores resultados en conjuntos de datos pequeños y ruidosos.
- La regresión logística es una opción adecuada cuando se requiere **interpretabilidad clínica**.
- Métricas como **recall y F1-score** son especialmente relevantes en problemas médicos.

---

## Trabajo futuro

- Uso de métricas adicionales (ROC-AUC, sensibilidad específica)
- Ingeniería de características clínicas
- Técnicas para tratar el desbalance de clases
- Ampliación del conjunto de datos

---

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Licencia

Este proyecto se utiliza con fines **académicos y educativos**.
