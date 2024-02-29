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
import random

data = pd.read_csv("sign_mnist_train.csv")
imagenes = data.drop(columns="label").values.reshape(-1,28,28)

#%%
### Analisis exploratorio ###

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

diccionario_letras = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

# Nos fijamos si los datos estan balanceados
# Contamos la cantidad de datos por cada etiqueta
cant_label = data["label"].value_counts().reset_index().sort_values("label")
plt.bar(cant_label['index'],cant_label['label'])
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.title("Cantidad de datos por cada etiqueta")


#%%
# Graficamos las imagenes sacando los bordes (tomando una imagen aleatoria por vez)
imagen_cortada = random.choice(imagenes)[6:22, 6:22]
plt.imshow(imagen_cortada, cmap="gray")
plt.title("Imagen recortada (16x16)")
plt.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)

#%%
# Aplicamos MaxPooling a las imagenes y todavia son identificables a simple vista
def max_pooling(image, pool_size=(2, 2)):
    height, width = image.shape
    pool_height, pool_width = pool_size

    new_height = height // pool_height
    new_width = width // pool_width

    pooled_image = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            region = image[i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width]
            pooled_image[i, j] = np.max(region)

    return pooled_image

nueva_imagen = max_pooling(random.choice(imagenes))
plt.imshow(nueva_imagen, cmap="gray")
plt.title("Imagenes con MaxPooling (14x14)")
plt.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)

#%%


#%%
# Diferencias entre E y L, o E y M
data_E = data[data["label"]==4].sample(frac=1).values[0][1:]
data_L = data[data["label"]==11].sample(frac=1).values[0][1:]
data_M = data[data["label"]==12].sample(frac=1).values[0][1:]

fig, axes = plt.subplots(2,2)
axes[0][0].imshow(data_E.reshape(28,28), cmap="gray")
axes[0][0].set_xlabel("Letra E")

axes[1][0].imshow(data_E.reshape(28,28), cmap="gray")
axes[1][0].set_xlabel("Letra E")

axes[0][1].imshow(data_L.reshape(28,28), cmap="gray")
axes[0][1].set_xlabel("Letra L")

axes[1][1].imshow(data_M.reshape(28,28), cmap="gray")
axes[1][1].set_xlabel("Letra M")

plt.subplots_adjust(hspace=0.2, wspace=-0.5)
for row in axes:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])

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
    
plt.subplots_adjust(hspace=0.2, wspace=-0.7)

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

# Declaramos las variables
n=3
X = data_A_L.drop('label', axis=1).sample(n, axis=1) #solo tomo n atributos de X
#X = data_A_L.drop("label",axis=1)[["pixel348","pixel320","pixel376", "pixel292", "pixel405"]]
y = data_A_L['label']


#Separamos en casos de train(80%) y test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=159)

# Declaramos el tipo de modelo
k=5
neigh = KNeighborsClassifier(n_neighbors = k)

# Entrenamos el modelo
neigh.fit(X_train, y_train)

# Evaluamos los resultados

accuracy = neigh.score(X_test, y_test)
print("Accuracy (test)", accuracy)

# Ahora vemos como varia la prediccion segun cuantos argumentos usemos
valores_n=range(1,20)

resultados_train=np.zeros(len(valores_n))
resultados_test=np.zeros(len(valores_n))

for n in valores_n:
    X = data_A_L.drop('label', axis=1).sample(n, axis=1)
    y = data_A_L['label']
    #Separamos en casos de train(80%) y test(20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Declaramos el tipo modelo
    k=5
    neigh=KNeighborsClassifier(n_neighbors=k)
    # Entrenamos el modelo
    neigh.fit(X_train,y_train)
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[n-1] = neigh.score(X_train, y_train)
    resultados_test[n-1]  = neigh.score(X_test , y_test )

# Graficamos R2 en funcion de n (para train y test)

plt.plot(valores_n, resultados_train, label = 'Train')
plt.plot(valores_n, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de KNN')
plt.xlabel('Cantidad de atributos')
plt.ylabel('R^2')
plt.xticks(valores_n)
plt.ylim(0.80,1.00)
