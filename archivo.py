            ### Laboratorio de datos: TP2 ###

## Analisis de imagenes con lenguaje de señas, y construccion ##
## de modelos predictivos para reconocimiento de imagenes     ##

## Integrantes: Manuel Gutman, Juan Cruz Mendoza y Kiara Yodko ##


# Importamos librerias y archivos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from inline_sql import sql, sql_val
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("sign_mnist_train.csv")


#%%

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

diccionario_letras = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 21:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

# Nos fijamos si los datos estan balanceados
# Contamos la cantidad de datos por cada etiqueta
cant_label = data["label"].value_counts().reset_index().sort_values("label")
plt.bar(data=cant_label, x="label", height="count")
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.title("Cantidad de datos por cada etiqueta")

#%%
# Construimos matriz de correlacion y tomamos aquellos pixeles que mas se relacionan con label
correlation_matrix = data.corr()["label"]
correlated_pixels = correlation_matrix[correlation_matrix > 0].sort_values(ascending=False)[1:]

#%%
# Diferencias entre las imagenes de una misma etiqueta (por ejemplo, C)
data_C = data[data["label"]==2].sample(frac=1).drop(columns=["label"])
filas = 5
columnas = 5
fig, axes = plt.subplots(filas,columnas)

for fila in range(filas):
    for columna in range(columnas):
        axes[fila][columna].imshow(data_C.values[fila+columna].reshape(28,28), cmap="gray")
        axes[fila][columna].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)  

#%%
### PUNTO 2 ###

# Construimos un nuevo dataframe solo con aquellas imagenes que sean una A o una L
consulta="""
SELECT *
FROM data
WHERE label==0 OR label==11
"""
data_A_L=sql^consulta

#◙ Vemos el numero total de muestras 
total_muestras_A_L=len(data_A_L)
print(total_muestras_A_L)

# Vemos si hay una cantidad parecida de señas A y L 

cant_A = data_A_L['label'].value_counts()[0]
cant_L = data_A_L['label'].value_counts()[11]

porcentaje_A=cant_A/total_muestras_A_L*100
print(porcentaje_A)
porcentaje_L=cant_L/total_muestras_A_L*100
print(porcentaje_L)

#declaramos las variables
n=5
X = data_A_L.drop('label', axis=1).sample(n, axis=1) #solo tomo n atributos de X
y = data_A_L['label']


#Separamos en casos de train(70%) y test(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=159)

# Declaramos el tipo de modelo
k=5
neigh = KNeighborsClassifier(n_neighbors = k)

# Entrenamos el modelo
neigh.fit(X_train, y_train)

# Evaluamos los resultados

r2 = neigh.score(X_test, y_test)
print("R^2 (test)", r2)
