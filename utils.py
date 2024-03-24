import numpy as np

# Funciones auxiliares para graficar
def elegir_imagen(letra, data):
    return data[data["label"]==letra].sample(1, axis=0).drop("label",axis=1).values.reshape(28,28)


def eliminar_ticks(axes):
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            
            
def eliminar_ticks_1D(axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        

def apilar_imagenes(letra, data):
    data_letra = data[data["label"]==letra].drop(columns=["label"]).values
    data_letra_prom = (data_letra.sum(axis=0) / len(data_letra)).reshape(28,28)
    return data_letra_prom


def pixels_max_contraste(cant_pixels, imagen_contrastada):

    # Balanceamos entre pixeleles positivos (blancos) y negativos (negros)
    if cant_pixels % 2 == 0:
        cant_pixels_positivos = cant_pixels //2
        cant_pixels_negativos = cant_pixels//2
    else:
        cant_pixels_positivos = cant_pixels//2 + 1
        cant_pixels_negativos = cant_pixels//2
        
    # Tomamos aquellos pixeles que mayor varianza tienen (mayor valor absoluto)
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