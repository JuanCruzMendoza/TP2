            ### Laboratorio de datos: TP2 ###

## Analisis de imagenes con lenguaje de señas, y construccion ##
## de modelos predictivos para reconocimiento de imagenes     ##

## Integrantes: Manuel Gutman, Juan Cruz Mendoza y Kiara Yodko ##


# Importamos librerias y archivos
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix 
import seaborn as sns
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz-10.01/bin/'
data = pd.read_csv("sign_mnist_train.csv")
imagenes = data.drop(columns="label").values.reshape(-1,28,28)

#%%
            ### Analisis exploratorio ###

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

diccionario_letras = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y"}

# Nos fijamos si los datos estan balanceados
# Contamos la cantidad de datos por cada etiqueta
cant_label = data["label"].value_counts().reset_index().sort_values("label")
plt.bar(cant_label['index'],cant_label['label'])
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.title("Cantidad de datos por cada etiqueta")


#%%
# Funciones para graficar
def elegir_imagen(letra, data):
    return data[data["label"]==letra].sample(1, axis=0).drop("label",axis=1).values.reshape(28,28)

def total_imagenes(letra, data):
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

def apilar_imagenes(letra, data):
    data_letra = data[data["label"]==letra].drop(columns=["label"]).values
    data_letra_prom = (data_letra.sum(axis=0) / len(data_letra)).reshape(28,28)
    return data_letra_prom

def pixels_max_contraste(cant_pixels, imagen_contrastada):

    # Tomamos pixeles tanto positivos (blancos) como negativos (negros)
    if cant_pixels % 2 == 0:
        cant_pixels_positivos = cant_pixels //2
        cant_pixels_negativos = cant_pixels//2
    else:
        cant_pixels_positivos = cant_pixels//2 + 1
        cant_pixels_negativos = cant_pixels//2
        
    # Tomamos aquellos pixeles que mayor contraste tienen (mayor valor absoluto)
    # Tomamos sus indices y devolvemos las coordenadas
    flattened_arr = imagen_contrastada.flatten()
    indices_negativos = np.argsort(flattened_arr)[:cant_pixels_negativos]
    indices_positivos = np.argsort(flattened_arr)[-cant_pixels_positivos:]
    coords_negativos = np.unravel_index(indices_negativos, imagen_contrastada.shape)
    coords_positivos = np.unravel_index(indices_positivos, imagen_contrastada.shape)
    coordenadas1 = list(zip(coords_negativos[0], coords_negativos[1]))
    coordenadas2 = list(zip(coords_positivos[0], coords_positivos[1]))
    
    coords_pixels = coordenadas1 + coordenadas2
    
    pixels_nombres = []
    for i,j in coords_pixels:
        pixels_nombres.append(f"pixel{i*28+j}")    
    
    return coords_pixels, pixels_nombres


#%% 
# Previsualizacion de los datos
filas = 4
columnas = 6
fig, axes = plt.subplots(filas,columnas)

for fila in range(filas):
    for columna in range(columnas):
        label = labels[fila*6+columna]
        axes[fila][columna].imshow(elegir_imagen(label, data), cmap="gray")
        axes[fila][columna].set_xlabel(diccionario_letras[label])
        
plt.subplots_adjust(hspace=0.5, wspace=-0.6)
plt.suptitle("Previsualizacion de las imagenes")
eliminar_ticks(axes)


#%%
# Graficamos las imagenes sacando los bordes (tomando una imagen aleatoria por vez)
imagen_cortada = elegir_imagen(random.choice(labels), data=data)[6:22, 6:22]
plt.imshow(imagen_cortada, cmap="gray")
plt.title("Imagen recortada (16x16)")
plt.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)


#%%
# Aplicamos MaxPooling a las imagenes y todavia son identificables a simple vista
fig, axes = plt.subplots(1,2)
nueva_imagen = elegir_imagen(2, data=data)
axes[1].imshow(max_pooling(nueva_imagen), cmap="gray")
axes[1].set_title("Imagen con MaxPooling (14x14)")

axes[0].imshow(nueva_imagen, cmap="gray")
axes[0].set_title("Imagen sin MaxPooling")

eliminar_ticks_1D(axes)


#%%
# Apilamos las imagenes de una letra y promediamos
# Podemos ver los pixeles mas importantes para caracterizar esa letra
fig, axes = plt.subplots(1,2)

axes[0].imshow(apilar_imagenes(2, data), cmap="gray")
axes[0].set_title("Imagenes apiladas de C")

axes[1].imshow(elegir_imagen(2, data),cmap="gray")
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
axes[0][0].imshow(elegir_imagen(4, data), cmap="gray")
axes[0][0].set_xlabel("Letra E")

axes[1][0].imshow(elegir_imagen(4, data), cmap="gray")
axes[1][0].set_xlabel("Letra E")

axes[0][1].imshow(elegir_imagen(11, data), cmap="gray")
axes[0][1].set_xlabel("Letra L")

axes[1][1].imshow(elegir_imagen(12, data), cmap="gray")
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
        axes[fila][columna].imshow(elegir_imagen(2, data), cmap="gray")
    
plt.subplots_adjust(hspace=0.2, wspace=-0.7)
eliminar_ticks(axes)


#%%
            ### PUNTO 2 ###

# Construimos un nuevo dataframe solo con aquellas imagenes que sean una A o una L
data_A_L = data[(data["label"]==0) | (data["label"]==11)]

# Vemos el numero total de muestras 
total_muestras_A_L=len(data_A_L)
print("Total de muestras: " + str(total_muestras_A_L))

# Vemos si hay una cantidad parecida de señas A y L 
cant_A = data_A_L['label'].value_counts()[0]
cant_L = data_A_L['label'].value_counts()[11]

porcentaje_A= round(cant_A/total_muestras_A_L*100, 1)
print("Porcentaje de A: " + str(porcentaje_A))
porcentaje_L= round(cant_L/total_muestras_A_L*100, 1)
print("Porcentaje de L: " + str(porcentaje_L))

# Vemos como son las señas
fig, axes = plt.subplots(1,2)
axes[0].imshow(elegir_imagen(0, data), cmap="gray")
axes[0].set_xlabel("Letra A")
axes[1].imshow(elegir_imagen(11, data), cmap="gray")
axes[1].set_xlabel("Letra L")
eliminar_ticks_1D(axes)

#%%
# Construyamos un modelo KNN con todos los atributos
X = data_A_L.drop('label', axis=1)
y = data_A_L['label']

#Separamos en casos de train(80%) y test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y, random_state=159)

# Declaramos el modelo de KNN
k=3
knn1= KNeighborsClassifier(n_neighbors = k)

# Entrenamos el modelo, con todos los atributos de X
knn1.fit(X_train, y_train)

# Evaluamos los resultados
accuracy = knn1.score(X_test, y_test)
print("Accuracy (test)", accuracy)


#%%
# Ahora vemos como varia la prediccion segun cuantos argumentos usemos, elegidos al azar
valores_n=range(1,20)
num_rep = 50

resultados_test  = np.zeros(( num_rep , len(valores_n)))
resultados_train = np.zeros(( num_rep , len(valores_n)))

for i in range(num_rep):
    for n in valores_n:
        X = data_A_L.drop('label', axis=1).sample(n, axis=1)
        y = data_A_L['label']
        #Separamos en casos de train(80%) y test(20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y)
        # Declaramos el tipo modelo
        k=3
        neigh=KNeighborsClassifier(n_neighbors=k)
        # Entrenamos el modelo
        neigh.fit(X_train,y_train)
        # Evaluamos el modelo con datos de train y luego de test
        resultados_train[i,n-1] = neigh.score(X_train, y_train)
        resultados_test[i,n-1]  = neigh.score(X_test , y_test )

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0) 

# Graficamos accuracy en funcion de n (para train y test)
plt.plot(valores_n, promedios_train, label = 'Train')
plt.plot(valores_n, promedios_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de KNN')
plt.xlabel('Cantidad de atributos (elegidos al azar)')
plt.ylabel('Accuracy')
plt.xticks(valores_n)
plt.grid()


#%%
# Veamos si podemos obtener resultados similares usando menos atributos
# Empecemos con PCA
X = data_A_L.drop('label', axis=1)
y = data_A_L['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y, random_state=159)

pca = PCA()
pca.n_components = 100 # Cantidad maxima de componentes
pca_X_train = pca.fit_transform(X_train)

# Veamos la varianza explicada de los datos en funciona de cuantos componentes tiene el PCA
varianza_explicada = pca.explained_variance_ / np.sum(pca.explained_variance_)
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

# Graficamos los datos segun sus dos componentes principales y su clasificacion
predicciones = np.hstack((pca_X_test, knn2.predict(pca_X_test).reshape(-1,1)))
predicciones_A = predicciones[predicciones[:,2]==0]
predicciones_L = predicciones[predicciones[:,2]==11]

# Predicciones que no fueron correctas
predicciones_error = predicciones[predicciones[:,2] != y_test]

plt.scatter(predicciones_A[:,0], predicciones_A[:,1], color="red")
plt.scatter(predicciones_L[:,0], predicciones_L[:,1], color="skyblue")
plt.scatter(predicciones_error[:,0], predicciones_error[:,1], color="black")

plt.legend(["A", "L", "Error"])
plt.title("KNN clasificacion con PCA")
plt.xlabel("1PC")
plt.ylabel("2PC")


#%%
# Probemos otro metodo: apilar las imagenes y contrastarlas
imagen_contrastada = apilar_imagenes(11, data)-apilar_imagenes(0, data)
plt.imshow(imagen_contrastada, cmap="gray")

# Encontramos los 2 pixels de mayor contraste y sus respectivos nombres
coords_pixels, pixels_nombres = pixels_max_contraste(2, imagen_contrastada)

for i,j in coords_pixels:
    plt.scatter(j, i, marker="+", color="red")

# Probamos el modelo utilizando los pixeles elegidos
X = data_A_L.drop('label', axis=1)[pixels_nombres]
y = data_A_L['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y, random_state=159)
k=3
knn3= KNeighborsClassifier(n_neighbors = k)
knn3.fit(X_train, y_train)

accuracy = knn3.score(X_test, y_test)
print("Accuracy (test)", accuracy)


#%%
# Encontramos los mejores k para cada cantidad de atributos, dado un mismo split del dataset
knn4 = KNeighborsClassifier()
X = data_A_L.drop("label", axis=1)
y = data_A_L["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y, random_state=159)

parametros = {"n_neighbors":range(1,50)}
cant_atributos = 5

for i in range(1,cant_atributos+1):    
    coords_pixels, pixels_nombres = pixels_max_contraste(i, imagen_contrastada)
    
    # Encuentra los mejores parametros de KNN (usando cross-validation, con 10 folds)
    grid_search = GridSearchCV(knn4, parametros, cv=10, scoring="accuracy")
    grid_search.fit(X_train[pixels_nombres], y_train)
    
    print("Cantidad de atributos: " + str(i))
    print("Mejor k: " + str(grid_search.best_params_["n_neighbors"]))
    print("Accuracy (train): "+ str(round(grid_search.best_score_,3)))
    print("Accuracy (test): " + str(round(grid_search.score(X_test[pixels_nombres], y_test),3)) + "\n")

#%%
# Elegimos 3 atributos, y graficamos accuracy sobre k
accuracy_train = []
accuracy_test = []
coords_pixels, pixels_nombres = pixels_max_contraste(3, imagen_contrastada)
rango_k = range(1,21)
for i in rango_k:
    knn5 = KNeighborsClassifier(n_neighbors= i)
    knn5.fit(X_train[pixels_nombres], y_train)
    accuracy_train.append(knn5.score(X_train[pixels_nombres], y_train))
    accuracy_test.append(knn5.score(X_test[pixels_nombres], y_test))
    
plt.plot(rango_k, accuracy_train, color="blue", label="Train")
plt.plot(rango_k, accuracy_test, color="orange", label="Test")
plt.legend()
plt.xticks(rango_k)
plt.grid()
plt.xlabel("Neighbors (K)")
plt.ylabel("Accuracy (test)")

#%%
# Elegimos 3 atributos, y graficamos accuracy sobre k, usando K-fold cross-validation
coords_pixels, pixels_nombres = pixels_max_contraste(1, imagen_contrastada)
rango_k = range(1,21)

resultados_test  = np.zeros(len(rango_k))
resultados_train = np.zeros(len(rango_k))

for i in rango_k:
    knn5 = KNeighborsClassifier(n_neighbors= i)
    knn5_cv = cross_validate(knn5, X=X_train[pixels_nombres], y=y_train,return_train_score=True, cv = 10, return_estimator=True)
    resultados_test[i-1] = knn5_cv["test_score"].mean()
    resultados_train[i-1] = knn5_cv["train_score"].mean()
    
plt.plot(rango_k, resultados_train, color="blue", label="Train")
plt.plot(rango_k, resultados_test, color="orange", label="Test")
plt.legend()
plt.xticks(rango_k)
plt.grid()
plt.xlabel("Neighbors (K)")
plt.ylabel("Accuracy (test)")
plt.title("Performance de KNN (3 atributos) con K-fold Cross-validation")

# Vemos que el mejor K es 3
knn5 = KNeighborsClassifier(n_neighbors=3)
knn5.fit(X_train[pixels_nombres], y_train)
print("Accuracy (evaluation set): ", knn5.score(X_test[pixels_nombres], y_test))


#%%
# Elegimos 3 atributos, y graficamos accuracy sobre k, usando K-fold cross-validation
# y cambiando los splits de los datasets

num_rep = 100
rango_k = range(1,11)
resultados_test  = np.zeros(( num_rep , len(rango_k)))
resultados_train = np.zeros(( num_rep , len(rango_k)))
X = data_A_L.drop('label', axis=1)
y = data_A_L['label']
coords_pixels, pixels_nombres = pixels_max_contraste(3, imagen_contrastada)

for i in range(num_rep):
    for k in rango_k:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y)
        neigh=KNeighborsClassifier(n_neighbors=k)

        neigh.fit(X_train[pixels_nombres],y_train)

        resultados_train[i,k-1] = neigh.score(X_train[pixels_nombres], y_train)
        resultados_test[i,k-1]  = neigh.score(X_test[pixels_nombres] , y_test )

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0) 

# Graficamos accuracy en funcion de n (para train y test)
plt.plot(rango_k, promedios_train, label = 'Train')
plt.plot(rango_k, promedios_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de KNN (3 atributos seleccionados)')
plt.xlabel('Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(rango_k)
plt.grid()


#%%
            ### PUNTO 3 ###

# Generamos el dataset que solo contiene vocales
vocales_data = data[data["label"].isin([0,4,8,14,20])]
vocales = ["A","E","I","O","U"]

# Vemos si estan balanceadas en cantidad 
cant_vocales = vocales_data["label"].value_counts().reset_index().sort_values("index")
plt.bar(vocales, cant_vocales["label"])
plt.ylabel("Contador")
plt.title("Cantidad de muestras por vocal")

#%%
#Declaramos las variables 
X = vocales_data.drop('label', axis=1)
y = vocales_data['label']

# Dividimos los datos en train(80%) y test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=159)

#Declaramos el modelo
arbol = DecisionTreeClassifier()
    
# Usamos stratify k folding y grid search para ver cual es el mejor modelo
skf = StratifiedKFold(n_splits=5)

rango_profundidades = range(1,21)
parametros_arbol = {"criterion":["gini","entropy"], "max_depth":rango_profundidades}
grid_search_arbol = GridSearchCV(arbol, cv=skf, param_grid=parametros_arbol, scoring="accuracy", return_train_score=True)
grid_search_arbol.fit(X_train, y_train)

#Nos quedamos con la que mejores resultados
mejor_profundidad= grid_search_arbol.best_params_["max_depth"]
mejor_criterio = grid_search_arbol.best_params_["criterion"]
print("Mejor profundidad:" ,mejor_profundidad)
print("Mejor criterio: ", mejor_criterio)

#%%
cv_results = grid_search_arbol.cv_results_

# Tomamos la accuracy promedio de los test folds (solo de entropy)
scores_entropy_test = cv_results["mean_test_score"][20:]
scores_entropy_train = cv_results["mean_train_score"][20:]

# Graficamos accuracy en funcion de la profundidad
plt.plot(rango_profundidades, scores_entropy_train, label="Train", color="blue")
plt.plot(rango_profundidades, scores_entropy_test, label="Test", color="orange")
plt.xlabel("Profundidad")
plt.ylabel("Accuracy")
plt.xticks(range(1,21))
plt.title("Performance de arbol de decision por profundidad")
plt.grid()

# Notamos que a partir de la profundidad 10, ya no hay mejora
# Usamos el mejor criterio del GridSearch, pero nos guiamos por el grafico para elegir depth=10
mejor_arbol = DecisionTreeClassifier(max_depth=10, criterion=mejor_criterio)

#Entrenamos al modelo
mejor_arbol.fit(X_train,y_train)

# Predecimos en el conjunto de test
y_pred_test = mejor_arbol.predict(X_test)

# Calculamos la precisión en el conjunto de evaluacion
acc_test = accuracy_score(y_test, y_pred_test)
print("Accuracy (set de evaluacion):", acc_test)


#%%
# Armamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_test)

# visualizacion la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=vocales, 
            yticklabels=vocales)
plt.xlabel('Prediccion')
plt.ylabel('Label')
plt.title('Matriz de Confusión')
plt.show()


#%%
# Graficamos el arbol 
fig = plt.figure( dpi = 500)
plot_tree(mejor_arbol, feature_names= X.columns,class_names= vocales,max_depth=3,
         filled=True, rounded=True)

fig.savefig("decistion_tree.png")

