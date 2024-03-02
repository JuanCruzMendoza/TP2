            ### Laboratorio de datos: TP2 ###

## Analisis de imagenes con lenguaje de señas, y construccion ##
## de modelos predictivos para reconocimiento de imagenes     ##

## Integrantes: Manuel Gutman, Juan Cruz Mendoza y Kiara Yodko ##


# Importamos librerias y archivos
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from inline_sql import sql, sql_val
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("sign_mnist_train.csv")
imagenes = data.drop(columns="label").values.reshape(-1,28,28)

#%%
### Analisis exploratorio ###

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

diccionario_letras = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y"}

# Nos fijamos si los datos estan balanceados
# Contamos la cantidad de datos por cada etiqueta
cant_label = data["label"].value_counts().reset_index().sort_values("label")
plt.bar(cant_label['index'],cant_label['label'])
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.title("Cantidad de datos por cada etiqueta")

#%%
# Funciones para graficar
def elegir_imagen(letra=random.choice(list(diccionario_letras.keys()))):
    return data[data["label"]==letra].sample(1, axis=0).drop("label",axis=1).values.reshape(28,28)

def total_imagenes(letra):
    return data[data["label"]==letra].drop("label",axis=1).values.reshape(-1,28,28)

def eliminar_ticks(axes):
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            
def eliminar_ticks_1D(axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
#%%
# Graficamos las imagenes sacando los bordes (tomando una imagen aleatoria por vez)
imagen_cortada = elegir_imagen()[6:22, 6:22]
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

fig, axes = plt.subplots(1,2)
nueva_imagen = elegir_imagen(2)
axes[0].imshow(max_pooling(nueva_imagen), cmap="gray")
axes[0].set_title("Imagen con MaxPooling (14x14)")

axes[1].imshow(nueva_imagen, cmap="gray")
axes[1].set_title("Imagen sin MaxPooling")

eliminar_ticks_1D(axes)


#%%
# Apilamos las imagenes de una letra y promediamos
# Podemos ver los pixeles mas importantes para caracterizar esa letra
fig, axes = plt.subplots(1,2)

def apilar_imagenes(letra):
    data_letra = data[data["label"]==letra].drop(columns=["label"]).values
    data_letra_prom = (data_letra.sum(axis=0) / len(data_letra)).reshape(28,28)
    return data_letra_prom

axes[0].imshow(apilar_imagenes(2), cmap="gray")
axes[0].set_title("Imagenes apiladas de C")

axes[1].imshow(elegir_imagen(2),cmap="gray")
axes[1].set_title("Imagen de muestra de C")

eliminar_ticks_1D(axes)


#%%
# Aplicamos PCA para reduccion de dimensionalidad
# Elegimos que pueda explicar el 90% de la varianza de los datos
pca = PCA(n_components = 0.9)
imagenes_pca = pca.fit_transform(imagenes.reshape(-1,784))

# Cantidad de atributos necesarios para explicar el 90% de la varianza
print(pca.n_components_)

# Podemos recrear cualquier imagen del dataset a partir de estos pocos atributos
reversed_imagen = pca.inverse_transform(imagenes_pca)
num_imagen = 0

fig, axes = plt.subplots(1,2)
axes[0].imshow(imagenes[num_imagen],cmap="gray")
axes[0].set_title("Imagen original")

axes[1].imshow(reversed_imagen[num_imagen].reshape(28,28), cmap="gray")
axes[1].set_title("Imagen con PCA")

eliminar_ticks_1D(axes)


#%%
# Diferencias entre E y L, o E y M
fig, axes = plt.subplots(2,2)
axes[0][0].imshow(elegir_imagen(4), cmap="gray")
axes[0][0].set_xlabel("Letra E")

axes[1][0].imshow(elegir_imagen(4), cmap="gray")
axes[1][0].set_xlabel("Letra E")

axes[0][1].imshow(elegir_imagen(11), cmap="gray")
axes[0][1].set_xlabel("Letra L")

axes[1][1].imshow(elegir_imagen(12), cmap="gray")
axes[1][1].set_xlabel("Letra M")

plt.subplots_adjust(hspace=0.2, wspace=-0.5)

eliminar_ticks(axes)


#%%
# Diferencias entre las imagenes de una misma etiqueta (por ejemplo, C)
filas = 5
columnas = 5
fig, axes = plt.subplots(filas,columnas)

for fila in range(filas):
    for columna in range(columnas):
        axes[fila][columna].imshow(elegir_imagen(2), cmap="gray")
    
plt.subplots_adjust(hspace=0.2, wspace=-0.7)

eliminar_ticks(axes)


#%%
### PUNTO 2 ###

# Construimos un nuevo dataframe solo con aquellas imagenes que sean una A o una L
data_A_L = data[(data["label"]==0) | (data["label"]==11)]

# Vemos el numero total de muestras 
total_muestras_A_L=len(data_A_L)
print(total_muestras_A_L)

# Vemos si hay una cantidad parecida de señas A y L 

cant_A = data_A_L['label'].value_counts()[0]
cant_L = data_A_L['label'].value_counts()[11]

porcentaje_A=cant_A/total_muestras_A_L*100
print(porcentaje_A)
porcentaje_L=cant_L/total_muestras_A_L*100
print(porcentaje_L)

#%%
# Declaramos las variables
X = data_A_L.drop('label', axis=1)
y = data_A_L['label']

#Separamos en casos de train(80%) y test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=159)

# Declaramos el modelo de KNN
k=3
knn1= KNeighborsClassifier(n_neighbors = k)

# Entrenamos el modelo, con todos los atributos de X
knn1.fit(X_train, y_train)

# Evaluamos los resultados
accuracy = knn1.score(X_test, y_test)
print("Accuracy (test)", accuracy)


#%%
# Veamos si podemos obtener resultados similares usando menos atributos
# Empecemos con PCA, viendo primero cuantos componentes usar
pca = PCA()
pca.n_components = 100 # Cantidad maxima de componentes
pca_X_train = pca.fit_transform(X_train)

# Veamos la varianza explicada de los datos en funciona de cuantos componentes tiene el PCA
varianza_explicada = pca.explained_variance_ / np.sum(pca.explained_variance_);
varianza_acumulada = np.cumsum(varianza_explicada)

fig, ax = plt.subplots()

ax.plot(varianza_acumulada, linewidth=2)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
ax.grid()
ax.set_xticks(np.arange(0,101,10))
ax.set_xlabel('Cantidad de componentes')
ax.set_ylabel('Varianza explicada')
ax.set_title("Aumento de la varianza segun componentes de PCA")


#%%
# Utilizamos PCA con 2 componentes
pca = PCA(n_components=2)
pca_X_train = pca.fit_transform(X_train)

k=3
knn2 = KNeighborsClassifier(n_neighbors = k)
knn2.fit(pca_X_train, y_train)

# Transformamos los datos test (sin hacer fit)
pca_X_test = pca.transform(X_test)

accuracy = knn2.score(pca_X_test, y_test)
print("Accuracy (test)", accuracy)


#%%
# Probemos otro metodo: apilar las imagenes y ver las diferencias entre si
imagen_contrastada = apilar_imagenes(11)-apilar_imagenes(0)
plt.imshow(imagen_contrastada, cmap="gray")

# Tomamos aquellos pixeles que mayor contraste tienen 
threshold = 85
pixels_mayores = np.where((imagen_contrastada < -74) | (imagen_contrastada > threshold))
coordenadas = list(zip(pixels_mayores[0], pixels_mayores[1]))

pixels_nombres = []
for i,j in coordenadas:
    pixels_nombres.append(f"pixel{i*28+j}")
    plt.scatter(j,i, marker="o", color="red")
    
# Probamos el modelo utilizando los pixeles elegidos
X = data_A_L.drop('label', axis=1)[pixels_nombres]
y = data_A_L['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=159)
k=3
knn3= KNeighborsClassifier(n_neighbors = k)
knn3.fit(X_train, y_train)

accuracy = knn3.score(X_test, y_test)
print("Accuracy (test)", accuracy)

#%%
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

# Graficamos exactitud en funcion de n (para train y test)

plt.plot(valores_n, resultados_train, label = 'Train')
plt.plot(valores_n, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de KNN')
plt.xlabel('Cantidad de atributos')
plt.ylabel('R^2')
plt.xticks(valores_n)
plt.ylim(0.80,1.00)

#%%
### PUNTO 3 ###

# Generamos el dataset que solo contiene vocales
vocales = data[(data["label"]==0) |
               (data["label"]==4) |
               (data["label"]==8) | 
               (data["label"]==14)|
               (data["label"]==20)]

# Vemos si estan balanceadas en cantidad 
cant_A = vocales['label'].value_counts()[0]
cant_E = vocales['label'].value_counts()[4] #Tiene menos que el resto, usamos stratify k fold
cant_I = vocales['label'].value_counts()[8]
cant_O = vocales['label'].value_counts()[14]
cant_U = vocales['label'].value_counts()[20]

#Declaramos las variables 
X = vocales.drop('label', axis=1)
y = vocales['label']

# Dividimos los datos en train(80%) y test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

# Evaluamos en distintas profundidades
df_profundidades = pd.DataFrame(columns=['Profundidad', 'Precisión en Train', 'Precisión en Test'])

for depth in range(1, 20):
    arbol = DecisionTreeClassifier(max_depth=depth)
    arbol.fit(X_train, y_train)
    y_pred_train = arbol.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    y_pred_test = arbol.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    df_profundidades = df_profundidades.append({'Profundidad': depth, 'Precisión en Train': acc_train, 'Precisión en Test': acc_test},ignore_index=True)
    
    
#Nos quedamos con la que mejores resultados de test tenga
mejor_profundidad=int((df_profundidades.sort_values(by='Precisión en Test', ascending=False)).iloc[0][0])
print("Mejor profundidad:" ,mejor_profundidad)

#Armamos el modelo usando esta profundidad
arbol1=DecisionTreeClassifier(max_depth=mejor_profundidad)

#Entrenamos al modelo
arbol1.fit(X_train,y_train)

# Usamos validación cruzada con Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(arbol1, X_train, y_train, cv=skf)
print("Precisión media CV:", cv_scores.mean())

# Predecimos en el conjunto de test
y_pred_test = arbol1.predict(X_test)

# Calculamos la precisión en el conjunto de test
acc_test = accuracy_score(y_test, y_pred_test)
print("Precisión en Test:", acc_test)

# Armamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_test)

# visualizacion la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['A', 'E', 'I', 'O', 'U'], yticklabels=['A', 'E', 'I', 'O', 'U'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()
    
