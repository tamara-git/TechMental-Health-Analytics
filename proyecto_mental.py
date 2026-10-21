"""
Analisis exploratorio del Dataset elegido: 
    https://www.kaggle.com/osmi/mental-health-in-tech-survey
"""

## Importo todas las librerías a utilizar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
## Nombramos y descargamos e}l dataset elegido

data_mental = pd.read_csv("survey.csv")

#%%=======================================
# Exploración
#=========================================


# Observamos la cantidad de columnas y filas
print("Dimensiones del dataset (filas, columnas):")
print(data_mental.shape)


# Analizamos el tipo de datos
print("Tipos de datos por columna:")
print(data_mental.dtypes)


#Columnas del dataset
print("Columnas del dataset):")
print(data_mental.columns)


#Observamos los datos nulos que posee el DataFrame
data_mental.isnull().sum()


#Observación de la cantidad de veces que se repite cada dato de la columna "Age"
data_mental.Age.value_counts()



#%%=======================================
# Preguntas
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



#%%============================
#%%=======================================
# Limpieza del dataset
#=========================================
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
data_mental.Gender.unique()

#%%=======================================
# Limpio los datos del sexo y los agrupo teniendo en cuenta 3 
# divisiones: "Female" (Femenino), "Male"(Masculino), 
# "NB" (No binario).
#=========================================
# 1. Limpiamos espacios al inicio/final y convertimos a string por seguridad
data_mental["Gender"] = data_mental["Gender"].astype(str).str.strip()


data_mental.Gender.replace({
    "M": "Male",
    "male": "Male",
    "female" : "Female",
    "m" : "Male",
    "Male-ish" : "Male",
    "maile" : "Male",
    "Trans-female" : "Trans-female",
    "Cis Female" : "Female",
    "F" : "Female",
    "something kinda male?" : "Male",
    "Cis Male" : "Male",
    "Woman" : "Female",
    "f" : "Female",
    "Mal" : "Male",
    "Male (CIS)" : "Male",
    "queer/she/they" : "Queer",
    "non-binary" : "NB",
    "Femake" : "Female",
    "woman" : "Female",
    "Make" : "Male",
    "Nah" : "Queer",
    "All" : "Queer",
    "Enby" : "NB",
    "fluid" : "Queer",
    "Genderqueer": "Queer",
    "Androgyne" : "NB",
    "Agender" : "NB",
    "cis-female/femme" : "Female",
    "Guy (-ish) ^_^" : "Male",
    "male leaning androgynous" : "NB",
    "Man" : "Male",
    "Trans woman" : "Trans-female",
    "msle" : "Male",
    "Neuter" : "NB",
    "Female (trans)" : "Male",
    "queer" : "Queer",
    "Female (cis)" : "Female",
    "Mail" : "Male",
    "cis male" : "Male",
    "A little about you" : "Queer",
     "Malr" : "Male",
     "p" : "Queer",
     "femail" : "Female",
     "Cis Man" : "Male",
     "ostensibly male, unsure what that really means" : "Male",
     "Male " : "Male",
     "Female " : "Female"

}, inplace = True)

#%%=======================================
## Identidades de género creadas
data_mental.Gender.value_counts()
#%%=======================================
#Reemplazo la columna Age por letras mínusculas usando el método rename y un diccionario de nombre viejo:nombre nuevo.
data_mental = data_mental.rename(columns={"Age":"age"}, inplace = True)
#%%=======================================
## Creo un filtro o máscara con la variable "age" (edad).
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


""" Usando el oneHotEncorder convierto las variables en categóricas con oneHotEncoder. Es decir, reemplazo las respuestas negativas (No) y positivas (Yes), y otras respuestas (Maybe, Some of them, etc), por ceros y unos._**"""

#%%=======================================
# Convierto la variable "mental_health_consequence" usando One-Hot Encoding
#========================================= 

# pd.get_dummies convierte la columna categórica en variables binarias y las une al DataFrame
data_mental = pd.get_dummies(data_mental, columns=["mental_health_consequence"], prefix="mental_health_consequence", dtype=int)
#%%=======================================
# Convierto la variable "family_history"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.family_history, "family_history_"))


data_mental_drop = data_mental.drop(columns="family_history", inplace = True)

#%%=======================================
# Convierto la variable "coworkers"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.coworkers, "coworkers_"))


data_mental_drop = data_mental.drop(columns="coworkers", inplace = True)

#%%=======================================
# Convierto la variable "treatment"
#=========================================  

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.treatment, "treatment_"))


data_mental_drop = data_mental.drop(columns="treatment", inplace = True)

#%%=======================================
# Convierto la variable "benefits"
#=========================================  

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.benefits, "benefits_"))


data_mental_drop = data_mental.drop(columns="benefits", inplace = True)

#%%=======================================
# Convierto la variable "seek_help"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.seek_help, "seek_help_"))


data_mental_drop = data_mental.drop(columns="seek_help", inplace = True)

#%%=======================================
# Convierto la variable "anonymity"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.anonymity, "anonymity_"))

data_mental_drop = data_mental.drop(columns="anonymity", inplace = True)

#%%=======================================
# Convierto la variable "obs_consequence"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.obs_consequence, "obs_consequence_"))


data_mental_drop = data_mental.drop(columns="obs_consequence", inplace = True)

#%%=======================================
# Convierto la variable "mental_vs_physical"
#========================================= 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.mental_vs_physical, "mental_vs_physical_"))


data_mental_drop = data_mental.drop(columns="mental_vs_physical", inplace = True)

#%%==============================================
# Convierto la variable "phys_healt_consequence"
#================================================ 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.phys_health_consequence, "phys_health_consequence_"))


data_mental_drop = data_mental.drop(columns="phys_health_consequence", inplace = True)

#%%============================================
# Convierto la variable "phys_healt_interview"
#==============================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.phys_health_interview, "phys_health_interview_"))


data_mental_drop = data_mental.drop(columns="phys_health_interview", inplace = True)

#%%========================================
# Convierto la variable "care_options"
#==========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.care_options, "care_options_"))


data_mental_drop = data_mental.drop(columns="care_options", inplace = True)

#%%========================================
# Convierto la variable "leave"
#==========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.leave, "leave_"))


data_mental_drop = data_mental.drop(columns="leave", inplace = True)

#%%================================================
# Convierto la variable "mental_health_interview"
#==================================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.mental_health_interview, "mental_health_interview_"))


data_mental_drop = data_mental.drop(columns="mental_health_interview", inplace = True)


#%%=========================================
# Convierto la variable "remote_work"
#===========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.remote_work, "remote_work_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="remote_work", inplace = True)


#%%=========================================
# Convierto la variable "no_employees"
#===========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.no_employees, "no_employees_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="no_employees", inplace = True)


#%%=========================================
# Convierto la variable "wellnes_program"
#===========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.wellness_program, "wellness_program_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="wellness_program", inplace = True)


#%%=========================================
# Convierto la variable "supervisor"
#===========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.supervisor, "supervisor_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="supervisor", inplace = True)


#%%=========================================
# Convierto la variable "tech_company"
#===========================================

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.tech_company, "tech_company_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="tech_company", inplace = True)


#%%=========================================
# Convierto la variable "Gender"
#=========================================== 

def oneHotEncoder(column, prefix):
  oneHot = pd.DataFrame()

  for value in column.unique():
    oneHot[prefix + value] = (column == value).astype("int")

  return oneHot

data_mental = data_mental.join(oneHotEncoder(data_mental.Gender, "Gender_"))

## Elimino la columna original.
data_mental_drop = data_mental.drop(columns="Gender", inplace = True)

## Marco los datos nulos de la variable "work_interfere" (Interferencia en el trabajo) como "Unanswered" (sin respuesta)
## Al ser la variable a predecir es imprescindible saber todos los datos que la compone y a su vez, los faltantes.

mask_interfere = data_mental["work_interfere"].isnull()
data_mental.loc[mask_interfere, ["work_interfere"]] = "Unanswered"



#%%=======================================
# DataFrame limpio
#=========================================
data_entrenamiento= data_mental

sns.countplot (x= "work_interfere", data=data_mental)

"""# Separación de entrenamiento y validación

Este paso es vital para probar el modelo. También debes de recordar, que en el caso de que quieras crear alguna variable extra o transformación al conjunto de datos (PCA por ejemplo) a usar en el entrenamiento, debes de hacer todos los cálculos sólo con el conjunto de entrenamiento, por lo que es probable que debas regresar al análisis exploratorio una vez realizada la división. Estos pasos los explicamos de una forma lineal, pero realmente los proyectos reales son iterativos y tienen mucho de ir y venir.
"""

X = data_mental.drop(columns="work_interfere")
y = data_mental["work_interfere"]

print(X.shape)
print(y.shape)

# Dividimos datos en train y test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=48)  #por default 25% de test

print("El tamaño de X_train es: ", X_train.shape)
print("El tamaño de X_test es: ",X_test.shape)
print("El tamaño de y_train es: ",y_train.shape)
print("El tamaño de y_test es: ",y_test.shape)

"""# Entrenamiento y prueba de modelos

En esta ocasión se responderán las preguntas planteadas, y expondrán sus resultados.

## Árboles de decisión
"""

#Importamos el modelo que utilizaremos: Arbol de decisión clasificación
from sklearn.tree import DecisionTreeClassifier

#Instanciamos el modelo que utilizaremos.
arbol = DecisionTreeClassifier()

#Entrenamos el modelo
arbol.fit(X_train, y_train)

from IPython.display import Image
from sklearn import tree

class_names = ["Rarely", "Often", "Sometimes", "Never", "Unanswered"]
plt.figure(figsize = (20,20))
tree.plot_tree(arbol, feature_names=data_mental.columns[:-1],filled=True,rounded=True, class_names=class_names)

## EVALUACIÓN DE DATOS DE TRAIN
#Vemos el resultado de la predicción de nuestro modelos sobre los datos utilizados para el entrenamiento
y_pred_train = arbol.predict(X_train)

#Utilizamos la métrica accuracy (exactitud)
from sklearn.metrics import accuracy_score

exactitud_train_arbol = accuracy_score(y_train, y_pred_train)
exactitud_train_arbol

#Podemos observar que el resultado de evaluar los datos de entrenamiento es pefecta, por lo cual podemos pensar que el modelo esta sobreajustado.

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

y_pred_arbol = arbol.predict(X_test)
#comparamos y_test(datos reales) con y_pred_arbol (datos predichos) para ver si el modelo hizo las predicciones correctas

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol)

from sklearn.metrics import confusion_matrix

matriz_arbol = confusion_matrix(y_test, y_pred_arbol)
# Vemos la matriz de confusión en una visualización heatmap

sns.heatmap(matriz_arbol, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#Metrica accuracy
from sklearn.metrics import accuracy_score

exactitud_arbol = accuracy_score(y_test, y_pred_arbol)
exactitud_arbol

"""#### Hiperparámetro distintos

"""

#Instanciamos el modelo que utilizaremos

arbol_depth3 = DecisionTreeClassifier(max_depth=3)

#Entrenamos el modelo
arbol_depth3.fit(X_train, y_train)

tree.plot_tree(arbol_depth3, feature_names=data_mental.columns[:-1],filled=True,rounded=True, class_names=class_names)
# EVALUACIÓN DE DATOS DE TRAIN
y_pred_train_depth3 = arbol_depth3.predict(X_train)
accuracy_score(y_train, y_pred_train_depth3)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_depth3 = arbol_depth3.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_depth3)
matriz_depth3 = confusion_matrix(y_test, y_pred_depth3)
sns.heatmap(matriz_depth3, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

exactitud_depth3 = accuracy_score(y_test, y_pred_depth3)
exactitud_depth3

"""Árbol con hiperparámetro depth=5."""

#Instanciamos el modelo que utilizaremos

arbol_depth5 = DecisionTreeClassifier(max_depth=5)

#Entrenamos el modelo
arbol_depth5.fit(X_train, y_train)

tree.plot_tree(arbol_depth5, feature_names=data_mental.columns[:-1],filled=True,rounded=True, class_names=class_names)

# EVALUACIÓN DE DATOS DE TRAIN
y_pred_train_depth5 = arbol_depth5.predict(X_train)
accuracy_score(y_train, y_pred_train_depth5)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_depth5 = arbol_depth5.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_depth4)
matriz_depth5 = confusion_matrix(y_test, y_pred_depth5)
sns.heatmap(matriz_depth5, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

exactitud_depth5 = accuracy_score(y_test, y_pred_depth5)
exactitud_depth5

"""## Resultados de árboles"""

#Comparamos la performance de los modelos

print("La exactitud del modelo árbol de decisión", round(exactitud_arbol,2))
print("La exactitud del modelo árbol de decisión con profundidad máxima de 3 es",round(exactitud_depth3,2))
print("La exactitud del modelo árbol de decisión con profundidad máxima de 5 es", round(exactitud_depth5,2))

"""## Random Forest"""

#Importamos el modelo
from sklearn.ensemble import RandomForestClassifier

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

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probamos nuestro modelo con los datos de test
y_pred_rf = random_forest.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol_completo)
from sklearn.metrics import confusion_matrix

matriz_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(matriz_rf, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#metrica accuracy
exactitud_rf = accuracy_score(y_test, y_pred_rf)
exactitud_rf

"""#### Hiperparámetro distinto

"""

#Instanciamos el modelo que utilizaremos
random_forest_2 = RandomForestClassifier(n_estimators = 500, max_depth = 2, max_features = 6, bootstrap = True)

#Entrenamos el modelo
random_forest_2.fit(X_train, y_train)

y_pred_train_rf2 = random_forest_2.predict(X_train)

accuracy_score(y_train, y_pred_train_rf2)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probamos nuestro modelo con los datos de test
y_pred_rf2 = random_forest_2.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol_completo)
matriz_rf2 = confusion_matrix(y_test, y_pred_rf2)
sns.heatmap(matriz_rf2, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#metrica accuracy
exactitud_rf2 = accuracy_score(y_test, y_pred_rf2)
exactitud_rf2

"""## Resultados del Random Forest"""

#Comparamos la performance de los modelos

print("La exactitud del modelo Random Forest es", round(exactitud_rf,2))
print("La exactitud del modelo Random Forest con profundidad máxima de 2 es", round(exactitud_rf2,2))

"""## Regresión logística
  Entreno un modelo de regresión logística.
"""

#importamos el modelo
from sklearn.linear_model import LogisticRegression

#Instanciamos el modelo que utilizaremos
reg_log = LogisticRegression()

#Entrenamos el modelo
reg_log.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_rl = reg_log.predict(X_train)
accuracy_score(y_train, y_pred_rl)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_rl = reg_log.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_rl)
matriz_rl = confusion_matrix(y_test, y_pred_rl)
sns.heatmap(matriz_rl, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#accuracy
exactitud_rl = accuracy_score(y_test, y_pred_rl)
exactitud_rl

"""## Resultado de Regresión Logística"""

#Modelo Regresión Logística

print("La exactitud del modelo Regresión Logística es", round(exactitud_rl,2))

"""## KNN
Entrenaremos un modelo de KNN: vecinos más cercanos con distintos hiperparámentros de cantidad de vecinos(n_neighbors)

"""

#importamos el modelo
from sklearn.neighbors import KNeighborsClassifier

#Instanciamos el modelo que utilizaremos n_neighbors=3
knn3= KNeighborsClassifier(n_neighbors=3 )

#Entrenamos el modelo
knn3.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_knn3 = knn3.predict(X_train)
accuracy_score(y_train, y_pred_knn3)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_knn3= knn3.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_knn3)
matriz_knn3= confusion_matrix(y_test, y_pred_knn3)
sns.heatmap(matriz_knn3, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#accuracy
exactitud_knn3 = accuracy_score(y_test, y_pred_knn3)
exactitud_knn3

"""##### Cambiando el hiperparámetro

cantidad de vecinos (n_neighbors): 5
"""

#Instanciamos el modelo que utilizaremos
knn5 = KNeighborsClassifier(n_neighbors=5)

#Entrenamos el modelo
knn5.fit(X_train, y_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_pred_knn5 = knn5.predict(X_train)
accuracy_score(y_train, y_pred_knn5)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_knn5 = knn5.predict(X_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_depth5)
matriz_knn5= confusion_matrix(y_test, y_pred_knn5)
sns.heatmap(matriz_knn5, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

exactitud_knn5 = accuracy_score(y_test, y_pred_knn5)
exactitud_knn5

"""## Resultados del modelo KNN"""

#Comparamos la performance de los modelos

print("La exactitud del modelo KNN con máximo de vecinos cercanos 3 es", round(exactitud_knn3,2))
print("La exactitud del modelo KNN con máximo de vecinos cercanos 5 es", round(exactitud_knn5,2))

"""#Estandarizaremos los datos para ver si el resultado es diferente

Importamos de la librería de Pre-procesamiento de Scikit-Learn el StandarScaler y creamos el objeto que utilizaremos
"""

# Creación de nuevos X_train y X_test para estandarizar los datos
X_std = data_mental.drop(columns="work_interfere")
y_std = data_mental["work_interfere"]

# Dividimos datos en train y test
from sklearn.model_selection import train_test_split

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, random_state=48)  #por default 25% de test

#Importamos de la librería de Pre-procesamiento de Scikit-Learn el StandarScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

"""##### Entrenamos y testeamos nuevamente

Utilizamos algunos modelos con los datos estandarizados

#####Utilizamos el modelo KNN con 3 vecinos más cercanos con los datos estandarizados
"""

#Instanciamos el modelo que utilizaremos

knn3_sca = KNeighborsClassifier(n_neighbors=3)

#Entrenamos el modelo con los datos de X_train escalados
knn3_sca.fit(X_train_std, y_train_std)

#Evaluación de datos de train
y_pred_knn3_sca = knn3_sca.predict(X_train_std)
accuracy_score(y_train, y_pred_knn3_sca)

"""Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probar nuestro modelo con los datos de test
y_pred_knn3_sca = knn3_sca.predict(X_test_std)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_depth2)
matriz_knn3_sca= confusion_matrix(y_test, y_pred_knn3_sca)
sns.heatmap(matriz_knn3_sca, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#accuracy
exactitud_knn3_sca = accuracy_score(y_test, y_pred_knn3_sca)
exactitud_knn3_sca

"""#####Utilizamos el modelo de arbol de decisión con los datos estandarizados"""

#Instanciamos el modelo que utilizaremos.
arbol_std = DecisionTreeClassifier()

#Entrenamos el modelo
arbol_std.fit(X_train_std, y_train_std)

from IPython.display import Image
from sklearn import tree

class_names = ["Rarely", "Often", "Sometimes", "Never", "Unanswered"]
plt.figure(figsize = (20,20))
tree.plot_tree(arbol_std, feature_names=data_mental.columns[:-1],filled=True,rounded=True, class_names=class_names)

## EVALUACIÓN DE DATOS DE TRAIN
#Vemos el resultado de la predicción de nuestro modelos sobre los datos utilizados para el entrenamiento
y_pred_arbol_std = arbol_std.predict(X_train_std)

#Utilizamos la métrica accuracy (exactitud)
from sklearn.metrics import accuracy_score

#Sobreajuste
exactitud_std = accuracy_score(y_train_std, y_pred_arbol_std)
exactitud_std

""" Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

y_pred_arbol_std = arbol_std.predict(X_test_std)
#comparamos y_test(datos reales) con y_pred_arbol (datos predichos) para ver si el modelo hizo las predicciones correctas

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol)

from sklearn.metrics import confusion_matrix

matriz_arbol_std = confusion_matrix(y_test_std, y_pred_arbol_std)
# Vemos la matriz de confusión en una visualización heatmap

sns.heatmap(matriz_arbol, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")


#Metrica accuracy
from sklearn.metrics import accuracy_score

exactitud_std = accuracy_score(y_test_std, y_pred_arbol_std)
exactitud_std

#Importamos el modelo
from sklearn.ensemble import RandomForestClassifier

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

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probamos nuestro modelo con los datos de test
y_pred_rf_std = random_forest_std.predict(X_test_std)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol_completo)
matriz_rf_std = confusion_matrix(y_test_std, y_pred_rf_std)
sns.heatmap(matriz_rf_std, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#metrica accuracy
exactitud_rf_std = accuracy_score(y_test_std, y_pred_rf_std)
exactitud_rf_std

"""##Resultado con datos estandarizados"""

#Comparamos la performance de los modelos

print("La exactitud del modelo KNN con datos estandarizados y un máximo de vecinos cercanos 3 es", round(exactitud_knn3_sca,2))
print("La exactitud del modelo arbol de decisión con datos estandarizados es", round(exactitud_std,2))
print("La exactitud del modelo Random Forest con datos estandarizados es", round(exactitud_rf_std,2))

"""#Dataframe 2"""

data_entrenamiento

## Marco los datos nulos de la variable "work_interfere" (Interferencia en el trabajo) como "Unanswered" (sin respuesta)

data_entrenamiento.dropna(subset= ["work_interfere"], inplace = True)

mask_entrenamiento = data_entrenamiento["work_interfere"] != "Unanswered"

data_entrenamiento= data_entrenamiento.loc[mask_entrenamiento]

data_entrenamiento

data_entrenamiento["work_interfere"].unique()

diccionario = {'Often': 'Often', 'Sometimes': 'Often','Never': 'Rarely', 'Rarely': 'Rarely'}
data_entrenamiento.replace(diccionario, inplace= True)

data_entrenamiento["work_interfere"].unique()

"""## Separación de entrenamiento y validación


"""

X_ent = data_entrenamiento.drop(columns="work_interfere")
y_ent = data_entrenamiento["work_interfere"]

print(X_ent.shape)
print(y_ent.shape)

# Dividimos datos en train y test

from sklearn.model_selection import train_test_split

X_ent_train, X_ent_test, y_ent_train, y_ent_test = train_test_split(X_ent, y_ent, random_state=48)  #por default 25% de test

print("El tamaño de X_train es: ", X_ent_train.shape)
print("El tamaño de X_test es: ",X_ent_test.shape)
print("El tamaño de y_train es: ",y_ent_train.shape)
print("El tamaño de y_test es: ",y_ent_test.shape)

"""## Entrenamiento y prueba de modelos

## Random Forest
"""

#Importamos el modelo
from sklearn.ensemble import RandomForestClassifier

#Instanciamos el modelo que utilizaremos
random_forest_ent = RandomForestClassifier()

#Entrenamos el modelo
random_forest_ent.fit(X_ent_train, y_ent_train)

#EVALUACIÓN DE DATOS DE TRAIN
y_ent_pred_train_rf = random_forest.predict(X_ent_train)

#Utilizaremos la métrica accuracy (exactitud)
exactitud_ent_rf = accuracy_score(y_ent_train, y_ent_pred_train_rf)
exactitud_ent_rf
# Observamos el sobreajuste

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probamos nuestro modelo con los datos de test
y_ent_pred_rf = random_forest.predict(X_ent_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol_completo)
from sklearn.metrics import confusion_matrix

matriz_ent_rf = confusion_matrix(y_ent_test, y_ent_pred_rf)
sns.heatmap(matriz_ent_rf, annot=True, fmt="d")
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#metrica accuracy
exactitud_ent_rf = accuracy_score(y_ent_test, y_ent_pred_rf)
exactitud_ent_rf

"""#### Hiperparámetro distinto

"""

#Instanciamos el modelo que utilizaremos
random_forest_ent_2 = RandomForestClassifier(n_estimators = 500, max_depth = 2, max_features = 6, bootstrap = True)

#Entrenamos el modelo
random_forest_ent_2.fit(X_ent_train, y_ent_train)

y_ent_pred_train_rf2 = random_forest_ent_2.predict(X_ent_train)

accuracy_score(y_ent_train, y_ent_pred_train_rf2)

"""##### Probamos y evaluamos nuestro modelo

Utilizamos el metodo predict para probar nuestro modelo con los datos de test. Luego comparamos la predicciones de nuestro modelo con el resultado real a través de una matrix de confusión y utilizando la métrica accuracy (exactitud)
"""

# Probamos nuestro modelo con los datos de test
y_ent_pred_rf2 = random_forest_ent_2.predict(X_ent_test)

# Matriz de confusión:comparando resultado original (y_test) con predicción del modelo (y_pred_arbol_completo)
matriz_ent_rf2 = confusion_matrix(y_ent_test, y_ent_pred_rf2)
sns.heatmap(matriz_ent_rf2, annot=True)
plt.xlabel("Etiquetas predichas")
plt.ylabel("Etiquetas reales")

#metrica accuracy
exactitud_ent_rf2 = accuracy_score(y_ent_test, y_ent_pred_rf2)
exactitud_ent_rf2

"""## Resultado de modelos con DataFrame 2"""

#Comparamos la performance de los modelos

print("La exactitud del modelo Random Forest en el nuevo DataFrame es", round(exactitud_ent_rf,2))
print("La exactitud del modelo Random Forest con máximo de profundidad 2, en el nuevo DataFrame es", round(exactitud_ent_rf2,2))