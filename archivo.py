            ### Laboratorio de datos: TP2 ###

## Analisis de imagenes con lenguaje de señas, y construccion ##
## de modelos predictivos para reconocimiento de imagenes     ##

## Integrantes: Manuel Gutman, Juan Cruz Mendoza y Kiara Yodko ##


# Importamos librerias y archivos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("sign_mnist_train.csv")


#%%

# Etiquetas (vemos que no existe la J=9 y Z=25)
labels = data["label"].unique()
labels.sort()

# Construimos matriz de correlacion y tomamos aquellos pixeles que mas se relacionan con label
correlation_matrix = data.corr()["label"]
correlated_pixels = correlation_matrix[correlation_matrix > 0].sort_values(ascending=False)[1:]

#%%

data_imagenes = data.drop(columns=["label"]).values

numero_imagen = 1
plt.imshow(data_imagenes[numero_imagen].reshape(28,28), cmap="gray")
    

