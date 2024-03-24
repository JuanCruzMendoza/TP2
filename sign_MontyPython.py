"""
            Laboratorio de datos: TP2 

    Analisis de imagenes con lenguaje de señas, 
    y construccion de modelos de clasificacion  

Integrantes: Manuel Gutman, Juan Cruz Mendoza y Kiara Yodko 
"""

# Importamos librerias y archivos
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 
from utils import elegir_imagen, eliminar_ticks, eliminar_ticks_1D, apilar_imagenes, pixels_max_contraste

# Nombrar a carpeta con la direccion del archivo sign_mnist_train.csv
carpeta = ""
data = pd.read_csv(carpeta + "sign_mnist_train.csv")
imagenes = data.drop(columns="label").values.reshape(-1,28,28)



#%%
            ### Analisis Exploratorio ###

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

# Cantidad de datos
cant_imagenes = len(data)

diccionario_letras = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

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
#plt.savefig("previsualizacion.png",dpi=200)
plt.show()


#%%
# Nos fijamos si los datos estan balanceados
# Contamos la cantidad de datos por cada etiqueta
cant_label = data.groupby(['label']).size().reset_index(name = 'cantidad_de_imagenes')

plt.bar(cant_label['label'], cant_label['cantidad_de_imagenes'], color="skyblue")

plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.xticks(range(0,26,1), labels=list(diccionario_letras.values()))
plt.title("Cantidad de datos por cada etiqueta")
#plt.savefig("cant_por_label.png",dpi=200)
plt.show()


#%%
# Diferencias entre E y L, o E y M
fig, axes = plt.subplots(1,3)

axes[0].imshow(elegir_imagen(4, data), cmap="gray")
axes[0].set_xlabel("Letra E")

axes[1].imshow(elegir_imagen(11, data), cmap="gray")
axes[1].set_xlabel("Letra L")

axes[2].imshow(elegir_imagen(12, data), cmap="gray")
axes[2].set_xlabel("Letra M")

plt.subplots_adjust(hspace=0.2, wspace=0.2)

eliminar_ticks_1D(axes)
#plt.savefig("dif_letras.png",dpi=200)
plt.show()


#%%
# Diferencias entre las imagenes de una misma etiqueta (por ejemplo, C)
filas = 4
columnas = 4
fig, axes = plt.subplots(filas,columnas)

for fila in range(filas):
    for columna in range(columnas):
        axes[fila][columna].imshow(elegir_imagen(2, data), cmap="gray")
    
plt.subplots_adjust(hspace=0.2, wspace=-0.7)
plt.suptitle("Diferencias entre letras C")

eliminar_ticks(axes)
#plt.savefig("dif_c.png",dpi=200)
plt.show()


#%%
# Apilamos las imagenes de una letra y promediamos
# Podemos ver los pixeles mas importantes para caracterizar esa letra
plt.imshow(apilar_imagenes(2, data), cmap="gray")

plt.title("Imagenes apiladas de C")
plt.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
#plt.savefig("c_apiladas.png",dpi=200)
plt.show()


#%%
# Aplicamos MaxPooling a las imagenes y todavia son identificables a simple vista
fig, axes = plt.subplots(1,2)
nueva_imagen = elegir_imagen(2, data=data)

pool_size = (2, 2)
imagen_pooled = block_reduce(nueva_imagen, pool_size, np.max)

axes[1].imshow(imagen_pooled, cmap="gray")
axes[1].set_title("Imagen con MaxPooling (14x14)")

axes[0].imshow(nueva_imagen, cmap="gray")
axes[0].set_title("Imagen original")

eliminar_ticks_1D(axes)
#plt.savefig("max_pooling.png",dpi=200)
plt.show()


#%%
# Aplicamos PCA para reduccion de dimensionalidad
# Elegimos que pueda explicar el 90% de la varianza de los datos
pca = PCA(n_components = 0.9)
imagenes_pca = pca.fit_transform(imagenes.reshape(-1,784))

# Cantidad de atributos necesarios para explicar el 90% de la varianza
print("Cantidad de componentes principales (90% varianza): ", pca.n_components_)

# Podemos recrear cualquier imagen del dataset a partir de estos pocos atributos
reversed_imagen = pca.inverse_transform(imagenes_pca)
num_imagen = 0

fig, axes = plt.subplots(1,2)
axes[0].imshow(imagenes[num_imagen],cmap="gray")
axes[0].set_title("Imagen original")

axes[1].imshow(reversed_imagen[num_imagen].reshape(28,28), cmap="gray")
axes[1].set_title("Imagen con PCA")

eliminar_ticks_1D(axes)
#plt.savefig("pca_muestra.png",dpi=200)
plt.show()



#%%
            ### Clasificacion Binaria (con KNN) ###

# Construimos un nuevo dataframe solo con aquellas imagenes que sean una A o una L
data_A_L = data[(data["label"]==0) | (data["label"]==11)]

# Vemos el numero total de muestras 
total_muestras_A_L=len(data_A_L)
print("Total de muestras: ", total_muestras_A_L)

# Vemos si hay una cantidad parecida de señas A y L 
cant_A = data_A_L['label'].value_counts()[0]
cant_L = data_A_L['label'].value_counts()[11]

porcentaje_A= round(cant_A/total_muestras_A_L*100, 1)
print("Porcentaje de A: " + str(porcentaje_A) + "%")

porcentaje_L= round(cant_L/total_muestras_A_L*100, 1)
print("Porcentaje de L: " + str(porcentaje_L)+ "%")

# Vemos como son las señas
fig, axes = plt.subplots(1,2)
axes[0].imshow(elegir_imagen(0, data), cmap="gray")
axes[0].set_xlabel("Letra A")
axes[1].imshow(elegir_imagen(11, data), cmap="gray")
axes[1].set_xlabel("Letra L")

eliminar_ticks_1D(axes)
#plt.savefig("letra_A_L.png",dpi=200)
plt.show()


#%%
# Construyamos un modelo KNN con todos los atributos
X = data_A_L.drop('label', axis=1)
y = data_A_L['label']

# Separamos datos de desarrollo (80%) y de evaluacion (20%)
# Siempre usaremos el mismo random_state
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state=159)

# Declaramos el modelo de KNN
knn1= KNeighborsClassifier(n_neighbors = 3)

# Entrenamos el modelo, con todos los atributos de X
knn1.fit(X_dev, y_dev)

# Evaluamos los resultados
knn1_accuracy = knn1.score(X_eval, y_eval)
print("Accuracy (evaluacion): " + str(knn1_accuracy*100) + "%")


#%%
# Ahora vemos como varia la prediccion segun cuantos atributos usemos, elegidos al azar
# Lo repetimos varias veces para promediar los resultados (bajar num_rep para hacerlo mas rapido)
rango_n = range(1,20)
num_rep = 10

resultados_test  = np.zeros((num_rep , len(rango_n)))
resultados_train = np.zeros((num_rep , len(rango_n)))

for i in range(num_rep):
    for n in rango_n:
        # Elegimos n pixeles aleatorios
        pixels_aleatorios = random.sample(list(X.columns), n)
        
        # Hacemos K-fold cross-validation 
        neigh = KNeighborsClassifier(n_neighbors= 3)
        neigh_cv = cross_validate(neigh, X=X_dev[pixels_aleatorios], y= y_dev ,return_train_score=True, cv = 10, return_estimator=True)
        
        # Guardamos el promedio de los resultados de los splits
        resultados_test[i,n-1] = neigh_cv["test_score"].mean()
        resultados_train[i,n-1] = neigh_cv["train_score"].mean()
     

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0) 

# Graficamos accuracy en funcion de n (para train y test)
# Se observa que a partir de 9 atributos aprox. se llega a un plateau
plt.plot(rango_n, promedios_train, label = 'Train')
plt.plot(rango_n, promedios_test, label = 'Test')

plt.legend()
plt.title('Performance del modelo de KNN')
plt.xlabel('Cantidad de atributos (elegidos al azar)')
plt.ylabel('Accuracy')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0,decimals=0))
plt.xticks(rango_n)
plt.grid()
#plt.savefig("acc_cant_n.png",dpi=200)
plt.show()



#%%
# Elegimos 9 atributos al azar y evaluamos la performance en funcion de K vecinos
rango_k = range(1,21)
num_rep = 10

resultados_test  = np.zeros((num_rep , len(rango_k)))
resultados_train = np.zeros((num_rep , len(rango_k)))

for i in range(num_rep):
    for k in rango_k:
        pixels_aleatorios = random.sample(list(X.columns), 9)
        
        neigh2 = KNeighborsClassifier(n_neighbors= k)
        neigh2_cv = cross_validate(neigh2, X=X_dev[pixels_aleatorios], y=y_dev,return_train_score=True, cv = 10, return_estimator=True)
        
        resultados_test[i,k-1] = neigh2_cv["test_score"].mean()
        resultados_train[i,k-1] = neigh2_cv["train_score"].mean()
    
promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0)     

plt.plot(rango_k, promedios_train, color="blue", label="Train")
plt.plot(rango_k, promedios_test, color="orange", label="Test")

plt.legend()
plt.xticks(rango_k)
plt.grid()
plt.xlabel("Neighbors (K)")
plt.ylabel("Accuracy")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
plt.title("Performance KNN (9 atributos aleatorios)")
#plt.savefig("acc_k_azar.png",dpi=200)
plt.show()


#%%
# Veamos si podemos obtener resultados similares usando menos atributos
# Empecemos con PCA
pca = PCA()
pca.n_components = 100 # Cantidad maxima de componentes
pca_X_dev = pca.fit_transform(X_dev)

# Veamos la varianza explicada de los datos en funcion de cuantos componentes tiene el PCA
varianza_explicada = pca.explained_variance_ / np.sum(pca.explained_variance_)
varianza_acumulada = np.cumsum(varianza_explicada)

plt.plot(varianza_acumulada, linewidth=2)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0,decimals=0))
plt.grid()
plt.xticks(np.arange(0,101,10))
plt.xlabel('Cantidad de componentes')
plt.ylabel('Varianza explicada')
plt.title("Aumento de la varianza segun componentes de PCA")
#plt.savefig("varianza.png",dpi=200)
plt.show()


#%%
# Utilizamos PCA con 2 componentes
pca = PCA(n_components=2)
pca_X_dev = pca.fit_transform(X_dev)

knn2 = KNeighborsClassifier(n_neighbors = 3)
knn2.fit(pca_X_dev, y_dev)

# Transformamos los datos de evaluacion (sin hacer fit)
pca_X_eval = pca.transform(X_eval)

accuracy_pca = knn2.score(pca_X_eval, y_eval)
print("Accuracy (evaluacion):" + str(accuracy_pca*100) + "%")

# Graficamos los datos segun sus dos componentes principales y su clasificacion
predicciones = np.hstack((pca_X_eval, knn2.predict(pca_X_eval).reshape(-1,1)))
predicciones_A = predicciones[predicciones[:,2]==0]
predicciones_L = predicciones[predicciones[:,2]==11]

# Predicciones que no fueron correctas
predicciones_error = predicciones[predicciones[:,2] != y_eval]

plt.scatter(predicciones_A[:,0], predicciones_A[:,1], color="red")
plt.scatter(predicciones_L[:,0], predicciones_L[:,1], color="skyblue")
plt.scatter(predicciones_error[:,0], predicciones_error[:,1], color="black")

plt.legend(["A", "L", "Error"])
plt.xlabel("1PC")
plt.ylabel("2PC")
plt.title("KNN clasificacion con PCA")
#plt.savefig("pca_clasificacion.png",dpi=200)
plt.show()


#%%
# Probemos otro metodo: apilar las imagenes y contrastarlas
imagen_contrastada = apilar_imagenes(11, data)-apilar_imagenes(0, data)

fig, ax = plt.subplots()
plt.imshow(imagen_contrastada, cmap="gray")
plt.title("Imagenes apiladas y contrastadas de A y L")

# Encontramos los 2 pixels de mayor contraste y sus respectivos nombres
coords_pixels, pixels_nombres = pixels_max_contraste(2, imagen_contrastada)

# Los graficamos
for i,j in coords_pixels:
    plt.scatter(j, i, marker="+", color="red")
    circulo = patches.Circle((j,i),1,color="red", alpha=0.2)
    ax.add_patch(circulo)
 
plt.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
#plt.savefig("A_L_apiladas.png",dpi=200)    
plt.show()

# Probamos un modelo utilizando los pixeles elegidos
knn3= KNeighborsClassifier(n_neighbors = 1)
knn3.fit(X_dev[pixels_nombres], y_dev)

accuracy_knn3 = knn3.score(X_eval[pixels_nombres], y_eval)
print("Accuracy (evaluacion): " + str(round(accuracy_knn3*100,2)) + "%")


#%%
# Encontramos los mejores k para cada cantidad de atributos
knn4 = KNeighborsClassifier()

parametros = {"n_neighbors":range(1,20)}
cant_atributos = 5

for i in range(1,cant_atributos+1):    
    coords_pixels, pixels_nombres = pixels_max_contraste(i, imagen_contrastada)
    
    # Encuentra los mejores parametros de KNN (usando K-fold cross-validation, con 10 folds)
    grid_search = GridSearchCV(knn4, parametros, cv=10, scoring="accuracy")
    grid_search.fit(X_dev[pixels_nombres], y_dev)
    
    print("Cantidad de atributos: " + str(i))
    print("Mejor K: " + str(grid_search.best_params_["n_neighbors"]))
    print("Accuracy (cv): "+ str(round(grid_search.best_score_*100,2)) + "%" + "\n")


#%%
# Elegimos 3 atributos, y graficamos accuracy sobre k, usando K-fold cross-validation
coords_pixels, pixels_nombres = pixels_max_contraste(3, imagen_contrastada)
rango_k = range(1,21)

resultados_test  = np.zeros(len(rango_k))
resultados_train = np.zeros(len(rango_k))

for i in rango_k:
    knn5 = KNeighborsClassifier(n_neighbors= i)
    knn5_cv = cross_validate(knn5, X=X_dev[pixels_nombres], y=y_dev,return_train_score=True, cv = 10, return_estimator=True)
    
    resultados_test[i-1] = knn5_cv["test_score"].mean()
    resultados_train[i-1] = knn5_cv["train_score"].mean()

plt.plot(rango_k, resultados_train, color="blue", label="Train")
plt.plot(rango_k, resultados_test, color="orange", label="Test")
plt.legend()
plt.xticks(rango_k)
plt.grid()
plt.xlabel("Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("Performance de KNN (3 atributos) con K-fold CV")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=1))
#plt.savefig("3_atributos.png",dpi=200)
plt.show()


#%%
# Vemos que el mejor K es 2
# Entrenamos el modelo con los datos de desarrollo y utilizamos el set de evaluacion para la accuracy
knn5 = KNeighborsClassifier(n_neighbors=2)
knn5.fit(X_dev[pixels_nombres], y_dev)
accuracy_dev = knn5.score(X_dev[pixels_nombres],y_dev)
accuracy_knn5 = knn5.score(X_eval[pixels_nombres], y_eval)

print("Accuracy (evaluacion): " + str(round(accuracy_knn5*100,2)) + "%")



#%%
            ### Clasificacion Multiclase (Arboles de decision) ###

# Generamos el dataset que solo contiene vocales
vocales_label = [0,4,8,14,20]
vocales_data = data[data["label"].isin(vocales_label)]
vocales = ["A","E","I","O","U"]

# Previsualizamos cada vocal
fig, axes = plt.subplots(1,5)
for i in range(5):
    axes[i].imshow(vocales_data[vocales_data["label"]==vocales_label[i]].drop("label",axis=1).values[0].reshape(28,28),cmap="gray")
    axes[i].set_xlabel(vocales[i])

eliminar_ticks_1D(axes)
#plt.savefig("previsualizacion_vocales.png",dpi=200)
plt.show()


#%%
# Vemos si las clases estan balanceadas (la E tiene menos que las demas)
cant_vocales = vocales_data["label"].value_counts().reset_index().sort_values("index")
plt.bar(vocales, cant_vocales["label"], color="skyblue")

plt.ylabel("Contador")
plt.title("Cantidad de muestras por vocal")
#plt.savefig("cant_vocales.png",dpi=200)
plt.show()


#%%
#Declaramos las variables 
X = vocales_data.drop('label', axis=1)
y = vocales_data['label']

# Dividimos los datos de desarrollo (80%) y evaluacion (20%)
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.2, random_state=159)

#Declaramos el modelo
arbol = DecisionTreeClassifier()
    
# Usamos Stratify K-fold para balancear las clases 
skf = StratifiedKFold(n_splits=5)

# Declaramos los posibles hiperparametros
rango_profundidades = range(1,21)
parametros_arbol = {"criterion":["gini","entropy"], "max_depth":rango_profundidades}

# Encontramos los mejores hiperparametros con Grid Search (usando K-fold cross-validation)
# (Nota: puede demorar, por lo que recomendamos disminuir n_splits o el rango de profundidades)
grid_search_arbol = GridSearchCV(arbol, cv=skf, param_grid=parametros_arbol, scoring="accuracy", return_train_score=True)
grid_search_arbol.fit(X_dev, y_dev)

# Nos quedamos con los mejores hiperparametros
mejor_profundidad= grid_search_arbol.best_params_["max_depth"]
mejor_criterio = grid_search_arbol.best_params_["criterion"]

print("Mejor profundidad:" ,mejor_profundidad)
print("Mejor criterio: ", mejor_criterio)


#%%
cv_results = grid_search_arbol.cv_results_

# Tomamos la accuracy promedio de los test folds (solo de entropy)
scores_entropy_eval = cv_results["mean_test_score"][len(rango_profundidades):]
scores_entropy_dev = cv_results["mean_train_score"][len(rango_profundidades):]

# Graficamos accuracy en funcion de la profundidad
plt.plot(rango_profundidades, scores_entropy_dev, label="Train", color="blue")
plt.plot(rango_profundidades, scores_entropy_eval, label="Test", color="orange")

plt.xlabel("Profundidad")
plt.ylabel("Accuracy")
plt.xticks(range(1,21))
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
plt.title("Performance del Arbol de Decision por profundidad")
plt.grid()
plt.legend()
#plt.savefig("acc_profundidad.png",dpi=200)
plt.show()

# Notamos que a partir de la profundidad 10, ya no hay mejora significativa
# Usamos el mejor criterio del GridSearch, pero nos guiamos por el grafico para elegir depth=10
mejor_arbol = DecisionTreeClassifier(max_depth=10, criterion=mejor_criterio)

#Entrenamos al modelo
mejor_arbol.fit(X_dev,y_dev)

# Predecimos en el conjunto de evaluacion
y_pred_eval = mejor_arbol.predict(X_eval)

# Calculamos la accuracy en el conjunto de evaluacion
accuracy_arbol = accuracy_score(y_eval, y_pred_eval)
print("Accuracy (evaluacion): " + str(round(accuracy_arbol*100,2)) + "%")


#%%
# Armamos la matriz de confusión
conf_matrix = confusion_matrix(y_eval, y_pred_eval)

# Visualizacion la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=vocales, 
            yticklabels=vocales)

plt.xlabel('Prediccion')
plt.ylabel('Label')
plt.title('Matriz de Confusion')
#plt.savefig("matriz_confusion.png",dpi=200)
plt.show()


#%%
# Por ultimo, el arbol de decision nos deja ver los pixeles que considero mas importantes
atributos_importantes = pd.DataFrame(mejor_arbol.feature_importances_,
                                index = X_dev.columns, columns=["importance"]).reset_index(names="pixeles")

# Tomamos los 10 pixeles que mas relevancia tuvieron en el entrenamiento
indices = atributos_importantes.nlargest(n=10, columns="importance").index
coords = np.unravel_index(indices, (28,28))

# Graficamos los pixeles en las imagenes de las vocales
fig, axes = plt.subplots(1,5, dpi=200)
for i in range(5):
    axes[i].imshow(vocales_data[vocales_data["label"]==vocales_label[i]].drop("label",axis=1).values[0].reshape(28,28),cmap="gray")
    axes[i].set_xlabel(vocales[i])
    axes[i].scatter(coords[1], coords[0], alpha=0.4)
    
eliminar_ticks_1D(axes)
#plt.savefig("pixeles_vocales.png",dpi=200)
plt.show()


