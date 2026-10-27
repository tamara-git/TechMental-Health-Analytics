"""
Analisis exploratorio del Dataset elegido: 
    https://www.kaggle.com/osmi/mental-health-in-tech-survey
"""
## Importo todas las librerías a utilizar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
#from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#%%=======================================
## Nombramos y descargamos el dataset elegido
data = pd.read_csv("survey.csv")
data_mental = data

#%%=======================================

#Observamos los datos nulos que posee el DataFrame
data_mental.isnull().sum()


#%%=======================================
#Observación de la cantidad de veces que se repite cada dato de la columna "Age"
data_mental.Age.value_counts()

#%%=======================================
# Preguntas generales
#=========================================

"""
1. ¿Qué información nos presenta la variable a predecir?
"""

# Realizamos un gráfico con Countplot para ver la variable a predecir
sns.countplot(x="work_interfere", data=data_mental, palette="colorblind")

plt.show()



"""
2.¿Hay una menor o mayor cantidad de personas con antecedentes familiares?
"""

#Graficamos con Countplot
sns.countplot(x= "family_history", data=data_mental, palette="colorblind")
plt.show()



"""
3.¿Hay grandes disparidades entre el numero de personas que recibieron tratamiento 
y las que no lo hicieron?
"""

#Graficamos la siguiente variable con Countplot
sns.countplot(x= "treatment", data=data_mental, palette="colorblind")



"""
4.¿Los empleados son todos de la misma nacionalidad? Si no es así, la encuesta
posee gran variedad de trabajadores con distintas nacionalidades o hay un país 
predominante en ella?
"""

# Al graficar la variable "Country" observamos que al poseer muchos datos no se pueden leer fácilmente
sns.countplot(x= "Country", data=data_mental)

# Para poder leerlo mejor, solo graficamos los países más predominantes en la encuesta
sns.countplot(
    x = "Country"[:30],
    data = data_mental[:30]
)

##Observación de la cantidad de veces que se repite cada dato de la columna "Country"
data_mental["Country"].value_counts()



"""
5.¿Cuántas personas trabajan de forma remota, fuera de una oficina?"""

sns.displot(data_mental.remote_work)


#%%=======================================
# Limpieza del dataset
#=========================================

#%%=======================================================
#Reemplazo la columna Age y Gender por letras mínusculas 
#=========================================================

data_mental = data_mental.rename(columns={"Age":"age"})
data_mental = data_mental.rename(columns={"Gender":"gender"})

#%%=======================================================
"""
Elimino las columnas que no son necesarias, ni tienen valor predictivo.

Elimino la variable "Self_employed" ya que además de tener datos nulos
 es una pregunta repetitiva a "Tech_company".
"""

data_mental.drop(columns = ["comments", "Timestamp", 
                            "state", "self_employed", "Country"],
                             inplace=True)

#%%=======================================
# Hay muchos datos anormales en la variable "Gender" (Género).
#=========================================
data_mental.gender.unique()

#%%=======================================
# Limpio los datos del sexo y los agrupo teniendo en cuenta 3 
# divisiones: "Female" (Femenino), "Male"(Masculino), 
# "NB" (No binario).
#=========================================
# 1. Limpiamos espacios al inicio/final y convertimos a string por seguridad
data_mental["gender"] = data_mental["gender"].astype(str).str.strip()

reemplazos = {
    "M": "Male",
    "male": "Male",
    "female" : "Female",
    "m" : "Male",
    "Male-ish" : "Male",
    "maile" : "Male",
    "Trans-female" : "Other",
    "Cis Female" : "Female",
    "F" : "Female",
    "something kinda male?" : "Male",
    "Cis Male" : "Male",
    "Woman" : "Female",
    "f" : "Female",
    "Mal" : "Male",
    "Male (CIS)" : "Male",
    "queer/she/they" : "Other",
    "non-binary" : "Other",
    "Femake" : "Female",
    "woman" : "Female",
    "Make" : "Male",
    "Nah" : "Other",
    "All" : "Other",
    "Enby" : "Other",
    "fluid" : "Other",
    "Genderqueer": "Other",
    "Androgyne" : "Other",
    "Agender" : "Other",
    "cis-female/femme" : "Female",
    "Guy (-ish) ^_^" : "Male",
    "male leaning androgynous" : "Other",
    "Man" : "Male",
    "Trans woman" : "Other",
    "msle" : "Male",
    "Neuter" : "Other",
    "Female (trans)" : "Other",
    "queer" : "Other",
    "Female (cis)" : "Female",
    "Mail" : "Male",
    "cis male" : "Male",
    "A little about you" : "Other",
     "Malr" : "Male",
     "p" : "Other",
     "femail" : "Female",
     "Cis Man" : "Male",
     "ostensibly male, unsure what that really means" : "Male",
}

data_mental["gender"] = data_mental["gender"].replace(reemplazos)

#=======================================
## Identidades de género creadas
data_mental.gender.value_counts()




#%%===========================================================
## Creo un filtro o máscara con la variable "age" (edad).
#=============================================================

mask_age = (data_mental["age"] <= 17) | (data_mental["age"] >=80)
mask_age

## Dicha máscara se compondrá por las edades erróneas del DataFrame.
data_age = data_mental.loc[mask_age]
data_age["age"]

##Muestra del filtro.
data_age

## Importo la librería random

import random

## Creo una lista para reemplazar aquellos valores defectuosos por unos posibles.

lista = list(range(18,60))

random.choices(list(range(18,60)),k=8)


data_mental.loc[mask_age, ["age"]] =random.choices(list(range(18,60)), k=8)


## Edades sustituidas
data_mental.loc[mask_age, ["age"]]

#%%=========================================
# Marco los datos nulos de la variable "work_interfere"
# (Interferencia en el trabajo) como "Unanswered" (sin respuesta)
#=========================================== 

"""
 Al ser la variable a predecir es imprescindible saber todos
 los datos que la compone y a su vez, los faltantes.
"""

mask_interfere = data_mental["work_interfere"].isnull()
data_mental.loc[mask_interfere, ["work_interfere"]] = "Unanswered"



#%%==================================================
# Convierto las variables usando One-Hot Encoding
#====================================================

# 1. Defino las columnas que quieres transformar
columnas_a_transformar = [
    "mental_health_consequence",
    "family_history",
    "coworkers",
    "treatment",
    "benefits",
    "seek_help",
    "anonymity",
    "obs_consequence",
    "mental_vs_physical",
    "phys_health_consequence",
    "phys_health_interview",
    "care_options",
    "leave",
    "mental_health_interview",
    "remote_work",
    "no_employees",
    "wellness_program",
    "supervisor",
    "tech_company",
    "gender"
]

# 2. Aplico One-Hot Encoding eliminando la primera categoría de cada una y 
data_mental = pd.get_dummies(
    data_mental, 
    columns=columnas_a_transformar, 
    drop_first=True, 
    dtype=int
)


#%%=====================================================
# DataFrame limpio - Ahora a realizar las predicciones
#=======================================================
"""Entrenamiento y prueba de modelos. """

data_entrenamiento= data_mental

sns.countplot (x= "work_interfere", data=data_mental)

# Separación de entrenamiento y validación

X = data_mental.drop(columns="work_interfere")
y = data_mental["work_interfere"]

print(X.shape)
print(y.shape)

# Dividimos datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=48)  #por default 25% de test

print("El tamaño de X_train es: ", X_train.shape)
print("El tamaño de X_test es: ",X_test.shape)
print("El tamaño de y_train es: ",y_train.shape)
print("El tamaño de y_test es: ",y_test.shape)


class_names = ["Rarely", "Often", "Sometimes", "Never", "Unanswered"]

#%%=============================================
# Árboles de decisión - Modelo de Clasificación
#===============================================

#Instanciamos el modelo que utilizaremos.
arbol = DecisionTreeClassifier()

#Entrenamos el modelo
arbol.fit(X_train, y_train)

plt.figure(figsize=(20, 20))
tree.plot_tree(
    arbol, 
    feature_names=X_train.columns,  # <--- Usar X_train.columns en lugar de data_mental
    filled=True, 
    rounded=True, 
    class_names=class_names
)
plt.show()

## EVALUACIÓN DE DATOS DE TRAIN
#Vemos el resultado de la predicción de nuestro modelos sobre los datos utilizados para el entrenamiento
y_pred_train = arbol.predict(X_train)

#Utilizamos la métrica accuracy (exactitud)
exactitud_train_arbol = accuracy_score(y_train, y_pred_train)
exactitud_train_arbol

"""Podemos observar que el resultado de evaluar los datos de entrenamiento es pefecta,
   por lo cual podemos pensar que el modelo esta sobreajustado."""

#%%========================

"""Probamos y evaluamos nuestro modelo.
   Utilizamos el metodo predict para probar nuestro modelo con los datos de test. 
   Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una
   matriz de confusión y utilizando la métrica accuracy (exactitud)
"""

y_pred_arbol = arbol.predict(X_test)

# comparamos y_test(datos reales) con y_pred_arbol (datos predichos) para ver si el modelo
# hizo las predicciones correctas representadas con una Matriz de confusión.

matriz_arbol = confusion_matrix(y_test, y_pred_arbol)
# Vemos la matriz de confusión en una visualización heatmap

sns.heatmap(matriz_arbol, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.show()

exactitud_arbol = accuracy_score(y_test, y_pred_arbol)
exactitud_arbol

#%%=============================================
# Probemos con Hiperparámetro distintos
#===============================================

#===============================================
# Árbol con hiperparámetro depth=3
#===============================================

#Instanciamos el modelo que utilizaremos
arbol_depth3 = DecisionTreeClassifier(max_depth=3)

#Entrenamos el modelo
arbol_depth3.fit(X_train, y_train)


plt.figure(figsize=(20, 20))
tree.plot_tree(
    arbol_depth3, 
    feature_names=X_train.columns,  # <--- Usar X_train.columns en lugar de data_mental
    filled=True, 
    rounded=True, 
    class_names=class_names
)
plt.show()

# EVALUACIÓN DE DATOS DE TRAIN
y_pred_train_depth3 = arbol_depth3.predict(X_train)
accuracy_score(y_train, y_pred_train_depth3)

#%%=============================================
"""Probamos y evaluamos nuestro modelo.
    Utilizamos el metodo predict para probar nuestro modelo con los datos de test. 
    Luego comparamos la predicciones de nuestro modelo con el resultado real a través
    de una matrix de confusión y utilizando la métrica accuracy (exactitud).
"""

# Probar nuestro modelo con los datos de test
y_pred_depth3 = arbol_depth3.predict(X_test)

# Vemos la matriz de confusión en una visualización heatmap
matriz_depth3 = confusion_matrix(y_test, y_pred_depth3)


sns.heatmap(matriz_depth3, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Árbol de Decisión depth3")
plt.show()


exactitud_depth3 = accuracy_score(y_test, y_pred_depth3)
exactitud_depth3

"""El árbol quedó demasiado limitado (con poca profundidad) y decidió que la forma más "fácil" de 
   minimizar errores fue ignorar por completo tres categorías enteras (Rarely, Often y Sometimes)."""

#%%==============================================
# Árbol con hiperparámetro depth=10
#================================================

#Instanciamos el modelo que utilizaremos

arbol_depth10 = DecisionTreeClassifier(max_depth=10)

#Entrenamos el modelo
arbol_depth10.fit(X_train, y_train)

plt.figure(figsize=(20, 20))
tree.plot_tree(
    arbol_depth10, 
    feature_names=X_train.columns,  # <--- Usar X_train.columns en lugar de data_mental
    filled=True, 
    rounded=True, 
    class_names=class_names
)
plt.show()


# EVALUACIÓN DE DATOS DE TRAIN
y_pred_train_depth10 = arbol_depth10.predict(X_train)
accuracy_score(y_train, y_pred_train_depth10)

#%%==============================================

""" Probamos y evaluamos nuestro modelo.
    Utilizamos el metodo predict para probar nuestro modelo con los datos de test.
    Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una 
    matriz de confusión y utilizando la métrica accuracy (exactitud).
"""

# Probar nuestro modelo con los datos de test
y_pred_depth10 = arbol_depth10.predict(X_test)

# Vemos la matriz de confusión en una visualización heatmap

matriz_depth10 = confusion_matrix(y_test, y_pred_depth10)


sns.heatmap(matriz_depth10, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Árbol de Decisión depth10")
plt.show()


exactitud_depth10 = accuracy_score(y_test, y_pred_depth10)
exactitud_depth10

#%%==============================================
# Resultados de las exactitudes de los árboles
#===============================================

#Comparamos la performance de los modelos

print("La exactitud del modelo árbol de decisión", round(exactitud_arbol,2))
print("La exactitud del modelo árbol de decisión con profundidad máxima de 3 es",round(exactitud_depth3,2))
print("La exactitud del modelo árbol de decisión con profundidad máxima de 10 es", round(exactitud_depth10,2))

#%%==============================================
# Random Forest
#===============================================

#Importamos el modelo

#Instanciamos el modelo que utilizaremos
random_forest = RandomForestClassifier()

#Entrenamos el modelo
random_forest.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_train_rf = random_forest.predict(X_train)

#Utilizaremos la métrica accuracy (exactitud)
exactitud_train_rf = accuracy_score(y_train, y_pred_train_rf)
exactitud_train_rf

# Observamos el sobreajuste

#%%==============================================
"""Probamos y evaluamos nuestro modelo
    Utilizamos el metodo predict para probar nuestro modelo con los datos de test.
    Luego comparamos la predicciones de nuestro modelo con el resultado real a través de 
    una matrix de confusión y utilizando la métrica accuracy (exactitud).
"""

# Probamos nuestro modelo con los datos de test
y_pred_rf = random_forest.predict(X_test)

# Vemos la matriz de confusión en una visualización heatmap
matriz_rf = confusion_matrix(y_test, y_pred_rf)


sns.heatmap(matriz_rf, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Random Forest")
plt.show()


#metrica accuracy
exactitud_rf = accuracy_score(y_test, y_pred_rf)
exactitud_rf

#%%==============================================
# Con hiperparámetros distintos
#===============================================

# Instanciamos el Random Forest equilibrando los pesos de las clases
random_forest_2 = RandomForestClassifier(n_estimators = 500, max_depth = 5, max_features = 6, bootstrap = True)

#Entrenamos el modelo
random_forest_2.fit(X_train, y_train)

y_pred_train_rf2 = random_forest_2.predict(X_train)

accuracy_score(y_train, y_pred_train_rf2)

# Vemos la matriz de confusión en una visualización heatmap

# Probamos nuestro modelo con los datos de test
y_pred_rf2 = random_forest_2.predict(X_test)

# Vemos la matriz de confusión en una visualización heatmap
matriz_rf2 = confusion_matrix(y_test, y_pred_rf2)


sns.heatmap(matriz_rf2, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Random Forest 500 estimators, max-depth5")
plt.show()


#metrica accuracy
exactitud_rf2 = accuracy_score(y_test, y_pred_rf2)
exactitud_rf2

#%%==============================================
# Resultados del Random Forest
#===============================================

#Comparamos la performance de los modelos

print("La exactitud del modelo Random Forest es", round(exactitud_rf,2))
print("La exactitud del modelo Random Forest con profundidad máxima de 5 es", round(exactitud_rf2,2))


#%%==============================================
# Modelo de Regresión logística
#===============================================

#Instanciamos el modelo que utilizaremos
reg_log = LogisticRegression()

#Entrenamos el modelo
reg_log.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_rl = reg_log.predict(X_train)
accuracy_score(y_train, y_pred_rl)

#===============================================

# Probar nuestro modelo con los datos de test
y_pred_rl = reg_log.predict(X_test)


# Vemos la matriz de confusión en una visualización heatmap
matriz_rl = confusion_matrix(y_test, y_pred_rl)

sns.heatmap(matriz_rl, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Logistic Regression")
plt.show()


#accuracy
exactitud_rl = accuracy_score(y_test, y_pred_rl)
exactitud_rl

#%%=============================================
# Resultados de Regresión Logística
#===============================================

print("La exactitud del modelo Regresión Logística es", round(exactitud_rl,2))

#%%=============================================
# Modelos de KNN - Vecinos más cercanos
#===============================================

#Instanciamos el modelo que utilizaremos n_neighbors=5
knn5= KNeighborsClassifier(n_neighbors=5 )

#Entrenamos el modelo
knn5.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_knn5 = knn5.predict(X_train)
accuracy_score(y_train, y_pred_knn5)

#===============================================
# Probamos nuestro modelo con los datos de test

y_pred_knn5= knn5.predict(X_test)


# Vemos la matriz de confusión en una visualización heatmap
matriz_knn5= confusion_matrix(y_test, y_pred_knn5)


sns.heatmap(matriz_knn5, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - KNN neighbors=5")
plt.show()


#accuracy
exactitud_knn5 = accuracy_score(y_test, y_pred_knn5)
exactitud_knn5

#%%=============================================
# Probando con otros hiperparámetros
#===============================================

#===============================================
# cantidad de vecinos (n_neighbors) = 10
#===============================================

#Instanciamos el modelo que utilizaremos
knn10 = KNeighborsClassifier(n_neighbors=10)

#Entrenamos el modelo
knn10.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_knn10 = knn10.predict(X_train)
accuracy_score(y_train, y_pred_knn10)

#===============================================

# Probamod nuestro modelo con los datos de test
y_pred_knn10 = knn10.predict(X_test)

# Vemos la matriz de confusión en una visualización heatmap
matriz_knn10 = confusion_matrix(y_test, y_pred_knn10)


sns.heatmap(matriz_knn10, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - KNN neighbors=10")
plt.show()


exactitud_knn10 = accuracy_score(y_test, y_pred_knn10)
exactitud_knn10

#%%================================
# Resultados del modelo KNN 
#==================================

#Comparamos la performance de los modelos

print("La exactitud del modelo KNN con máximo de vecinos cercanos 5 es", round(exactitud_knn5,2))
print("La exactitud del modelo KNN con máximo de vecinos cercanos 10 es", round(exactitud_knn10,2))


#%%===================================
# Modelos con datos estandarizados
#=====================================
"""
    Estandarizaremos los datos para ver si el resultado es diferente.
    Importamos de la librería de Pre-procesamiento de Scikit-Learn el StandarScaler y
    creamos el objeto que utilizaremos
"""

# Creación de nuevos X_train y X_test para estandarizar los datos
X_std = data_mental.drop(columns="work_interfere")
y_std = data_mental["work_interfere"]

# Dividimos datos en train y test

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, random_state=48)  #por default 25% de test

scaler = StandardScaler()

"""Entrenamos y testeamos nuevamente.

   Utilizamos algunos modelos con los datos estandarizados
   Utilizamos el modelo KNN con 3 vecinos más cercanos con los datos estandarizados
"""

#%%===================================
# KNN con datos estandarizados
#=====================================
#Instanciamos el modelo que utilizaremos

knn5_sca = KNeighborsClassifier(n_neighbors=5)

#Entrenamos el modelo con los datos de X_train escalados
knn5_sca.fit(X_train_std, y_train_std)

#Evaluación de datos de train
y_pred_knn5_sca = knn5_sca.predict(X_train_std)
accuracy_score(y_train, y_pred_knn5_sca)

#=====================================

# Probar nuestro modelo con los datos de test
y_pred_knn5_sca = knn5_sca.predict(X_test_std)

# Vemos la matriz de confusión en una visualización heatmap
matriz_knn5_sca= confusion_matrix(y_test, y_pred_knn5_sca)


sns.heatmap(matriz_knn5_sca, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - KNN estandarizado neighbors=5")
plt.show()


#accuracy
exactitud_knn5_sca = accuracy_score(y_test, y_pred_knn5_sca)
exactitud_knn5_sca

#%%===========================================
# Árbol de decisión con datos estandarizados
#=============================================

#Instanciamos el modelo que utilizaremos.
arbol_std = DecisionTreeClassifier()

#Entrenamos el modelo
arbol_std.fit(X_train_std, y_train_std)

plt.figure(figsize=(20, 20))
tree.plot_tree(
    arbol_std, 
    feature_names=X_train.columns,  
    filled=True, 
    rounded=True, 
    class_names=class_names
)
plt.show()

## EVALUACIÓN DE DATOS DE TRAIN
#Vemos el resultado de la predicción de nuestro modelos sobre los datos utilizados para el entrenamiento
y_pred_arbol_std = arbol_std.predict(X_train_std)

#Utilizamos la métrica accuracy (exactitud)

#Sobreajuste
exactitud_std = accuracy_score(y_train_std, y_pred_arbol_std)
exactitud_std

#%%===========================================
# Probar nuestro modelo con los datos de test
#=============================================

y_pred_arbol_std = arbol_std.predict(X_test_std)

# Vemos la matriz de confusión en una visualización heatmap
matriz_arbol_std = confusion_matrix(y_test_std, y_pred_arbol_std)


sns.heatmap(matriz_arbol_std, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Árbol de decisión, neighbors=5")
plt.show()

#Metrica accuracy
exactitud_std = accuracy_score(y_test_std, y_pred_arbol_std)
exactitud_std

#%%===========================================
# Random Forest con datos estandarizados
#=============================================

#Instanciamos el modelo que utilizaremos
random_forest_std = RandomForestClassifier()

#Entrenamos el modelo
random_forest_std.fit(X_train_std, y_train_std)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_rf_std = random_forest_std.predict(X_train_std)

#Utilizaremos la métrica accuracy (exactitud)
exactitud_rf_std = accuracy_score(y_train_std, y_pred_rf_std)
exactitud_rf_std
# Observamos el sobreajuste

#%%===========================================

#%%=============================================
# Probamos nuestro modelo con los datos de test
#===============================================
y_pred_rf_std = random_forest_std.predict(X_test_std)

# Vemos la matriz de confusión en una visualización heatmap
matriz_rf_std = confusion_matrix(y_test_std, y_pred_rf_std)


sns.heatmap(matriz_rf_std, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Random Forest estandarizado")
plt.show()

#metrica accuracy
exactitud_rf_std = accuracy_score(y_test_std, y_pred_rf_std)
exactitud_rf_std


#%%===========================================
# Resultado con datos estandarizados
#=============================================


#Comparamos la performance de los modelos

print("La exactitud del modelo KNN con datos estandarizados y un máximo de vecinos cercanos 5 es", round(exactitud_knn5_sca,2))
print("La exactitud del modelo arbol de decisión con datos estandarizados es", round(exactitud_std,2))
print("La exactitud del modelo Random Forest con datos estandarizados es", round(exactitud_rf_std,2))

"""La etiqueta 'Never' sesga los datos, pues fue la respuesta más frecuente en el dataframe."""

#%%===========================================
# Dataframe con work_interfere modificado
#=============================================

data_entrenamiento

# Marco los datos nulos de la variable "work_interfere" (Interferencia en el trabajo) como "Unanswered" (sin respuesta).

data_entrenamiento.dropna(subset= ["work_interfere"], inplace = True)

mask_entrenamiento = data_entrenamiento["work_interfere"] != "Unanswered"

data_entrenamiento= data_entrenamiento.loc[mask_entrenamiento]

data_entrenamiento

data_entrenamiento["work_interfere"].unique()

diccionario = {'Often': 'Often', 'Sometimes': 'Often','Never': 'Rarely', 'Rarely': 'Rarely'}
data_entrenamiento.replace(diccionario, inplace= True)

data_entrenamiento["work_interfere"].unique()

class_names2 = [ "Often", "Never"]
#%%===========================================
# Separación de entrenamiento y validación.

X_ent = data_entrenamiento.drop(columns="work_interfere")
y_ent = data_entrenamiento["work_interfere"]

print(X_ent.shape)
print(y_ent.shape)

# Dividimos datos en train y test

X_ent_train, X_ent_test, y_ent_train, y_ent_test = train_test_split(X_ent, y_ent, random_state=48)  #por default 25% de test

print("El tamaño de X_train es: ", X_ent_train.shape)
print("El tamaño de X_test es: ",X_ent_test.shape)
print("El tamaño de y_train es: ",y_ent_train.shape)
print("El tamaño de y_test es: ",y_ent_test.shape)

#%%================================================
# Entrenamiento y prueba de modelos Random Forest
#==================================================

#Importamos el modelo

#Instanciamos el modelo que utilizaremos
random_forest_ent = RandomForestClassifier()

#Entrenamos el modelo
random_forest_ent.fit(X_ent_train, y_ent_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_ent_pred_train_rf = random_forest_ent.predict(X_ent_train)

#Utilizaremos la métrica accuracy (exactitud)
exactitud_ent_rf = accuracy_score(y_ent_train, y_ent_pred_train_rf)
exactitud_ent_rf
# Observamos el sobreajuste

#%%=============================================
# Probamos nuestro modelo con los datos de test
#===============================================

y_ent_pred_rf = random_forest_ent.predict(X_ent_test)

#Vemos la matriz de confusión en una visualización heatmap
matriz_ent_rf = confusion_matrix(y_ent_test, y_ent_pred_rf)

sns.heatmap(matriz_ent_rf, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names2, 
            yticklabels=class_names2)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión - Random Forest - dataframe modificado")
plt.show()


#metrica accuracy
exactitud_ent_rf = accuracy_score(y_ent_test, y_ent_pred_rf)
exactitud_ent_rf

#%%===========================================
# Probamos el modelo con otro hiperparámetro
#=============================================

#Instanciamos el modelo que utilizaremos
random_forest_ent_2 = RandomForestClassifier(n_estimators = 500, max_depth = 5, max_features = 6, bootstrap = True)

#Entrenamos el modelo
random_forest_ent_2.fit(X_ent_train, y_ent_train)

y_ent_pred_train_rf2 = random_forest_ent_2.predict(X_ent_train)

accuracy_score(y_ent_train, y_ent_pred_train_rf2)

#%%=============================================
# Probamos nuestro modelo con los datos de test
#===============================================

y_ent_pred_rf2 = random_forest_ent_2.predict(X_ent_test)

#Vemos la matriz de confusión en una visualización heatmap
matriz_ent_rf2 = confusion_matrix(y_ent_test, y_ent_pred_rf2)


sns.heatmap(matriz_ent_rf2, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names2, 
            yticklabels=class_names2)

plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

plt.title("Matriz de Confusión - Random Forest2 - dataframe modificado")
plt.show()


#metrica accuracy
exactitud_ent_rf2 = accuracy_score(y_ent_test, y_ent_pred_rf2)
exactitud_ent_rf2

#%%=============================================
# Resultados de modelos con DataFrame modificado
#===============================================

#Comparamos la performance de los modelos

print("La exactitud del modelo Random Forest en el nuevo DataFrame es", round(exactitud_ent_rf,2))
print("La exactitud del modelo Random Forest con máximo de profundidad 5, en el nuevo DataFrame es", round(exactitud_ent_rf2,2))


#%%=============================================
# Reporte de clasificación
#===============================================

print(classification_report(y_ent_test, y_ent_pred_rf2, target_names=sorted(y_ent.unique())))

#%%=============================================
# Validación cruzada
#===============================================
scores = cross_val_score(random_forest_ent_2, X_ent, y_ent, cv=5, scoring='accuracy')
print("Exactitudes por pliegue:", scores)
print("Exactitud promedio:", scores.mean())


#%%=============================================
# Dataframe con resultados
#===============================================
# Creamos un DataFrame que una los datos de test con la realidad y la predicción
df_resultados = pd.DataFrame({
    'Real': y_ent_test,
    'Prediccion': y_ent_pred_rf2
})

# Vemos las primeras filas
print(df_resultados.head(10))
