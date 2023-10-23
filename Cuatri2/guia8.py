# Ej 1
def contarlineas(nombre_archivo:str)->int:
    archivo = open(nombre_archivo, "r")
    lineas = archivo.readlines()
    archivo.close()
    return len(lineas)

def existePalabra(palabra:str,nombre_archivo:str)->bool:
    archivo = open(nombre_archivo,"r")
    texto = archivo.read()
    archivo.close()
    return palabra in texto

def cantidadApariciones(palabra:str, nombre_archivo:str)->int:
    archivo = open(nombre_archivo,"r")
    texto:str = archivo.read()
    lista_palabras: list[str] = []
    palabra_nueva:str = ""
    cantidad:int = 0
    for i in texto:
        if i.lower() >= "a" and i.lower() <= "z":
            palabra_nueva += i
        elif i == " " or i == "\n":
            lista_palabras.append(palabra_nueva)
            palabra_nueva = ""
    
    lista_palabras.append(palabra_nueva)
    
    for i in lista_palabras:
        if palabra == i:
            cantidad += 1

    archivo.close()
    return cantidad


# Ej 2
def clonarSinComentarios(nombre_archivo:str):
    def es_un_comentario(line:str)->bool:
        for c in line:
            if c != " ":
                if c == "#":
                    return True
                return False
        return False
    
    archivo = open(nombre_archivo, "r")
    arch_sin_comentarios = open("clonadoSinComentarios.txt", "w")

    lineas = archivo.readlines()
    for linea in lineas:
        if not es_un_comentario(linea):
            arch_sin_comentarios.write(linea)
    archivo.close()
    arch_sin_comentarios.close()


# Ej 3
def texto_reverso(nombre_archivo:str):
    archivo = open(nombre_archivo,"r")
    archivo_nuevo = open("archivo_reverso.txt","w")
    lineas: list[str] = archivo.readlines()
    for i in range(len(lineas)-1,-1,-1):
        
        archivo_nuevo.write(lineas[i])
        if i == len(lineas)-1:
            archivo_nuevo.write("\n")
    archivo.close()

# Ej 4
def frase_final(nombre_archivo:str,frase:str):
    archivo = open(nombre_archivo,"r")
    texto = archivo.read()
    archivo = open(nombre_archivo,"w")
    archivo.write(texto+frase)
    archivo.close()



# Ej 5
def frase_inicial(nombre_archivo:str,frase:str):
    archivo = open(nombre_archivo,"r")
    texto = archivo.read()
    archivo = open(nombre_archivo,"w")
    archivo.write(frase+texto)
    archivo.close()


# Ej 6
def leer_binario(nombre_archivo:str)->"list[str]":
    archivo = open(nombre_archivo,"rb")
    texto:str = ""
    for i in archivo.read():
        texto += chr(i)
    pos_palabra:str=""
    lista_palabras: list[str] = []
    for i in texto:
        if (i.lower() >= "a" and i.lower() <= "z") or i == " " or i == "_":
            pos_palabra += i
        else:
            if len(pos_palabra) >= 5:
                lista_palabras.append(pos_palabra)
                pos_palabra = ""
    if pos_palabra != "":
        lista_palabras.append(pos_palabra)
    archivo.close()
    return lista_palabras


# Ej 7
def promedioEstudiante(lu : str)->float:
    archivo = open(lu,"r")
    notas:list[float] = []
    lineas = archivo.readlines()
    for line in lineas:
        for i in range(len(line)):
            if line == lineas[-1]:
                line+=" "
            if line[i:i+5] == "nota ":
                notas.append(float(line[i+6:-2]))
                break
    
    promedio:float = 0
    for i in range(len(lineas)):
        promedio += notas[i]
    promedio = promedio / len(lineas)

    return promedio


# Ej 8
from queue import LifoQueue
import random

def copiar(p:LifoQueue)->LifoQueue:
    elements: [int] = []
    while not p.empty():
        elements.append(p.get())
    p_copy: LifoQueue = LifoQueue()
    for i in range(len(elements)-1,-1,-1):
        p.put(elements[i])
        p_copy.put(elements[i])
    return p_copy

def generar_nros_al_azar(n:int, desde:int,hasta:int)->LifoQueue:
    pila = LifoQueue()
    for i in range(n):
        pila.put(random.randint(desde,hasta))
    return pila


# Ej 9
def cantidad_elementos(p: LifoQueue)->int:
    p_copy: LifoQueue = copiar(p)
    contador:int = 0
    while not p_copy.empty():
        p_copy.get()
        contador+=1
    return contador


# Ej 10
# Ojo: tiene que ser in, no podemos modificar la pila


def buscarElMaximo(p:LifoQueue)->int:
    p_copy: LifoQueue = copiar(p)
    value = p_copy.get()
    while not p.empty():
        next_value = p_copy.get()
        value = max(next_value, value)
    return value


# Ej 16
import random
from queue import Queue
def armarSecuenciaDeBingo()->"Queue[int]":
    l:[int] = []
    for i in range(0,99+1,1):
        l.append(i)
    random.shuffle(l)
    result: Queue[int] = Queue()
    for elem in l:
        result.put(elem)
    return result

def jugarCartonBingo(carton:"list[int]", cola:"Queue[int]")->int:
    for i in range(100):
        numero:int = cola.get()
        contador:int = 0
        if numero in carton:
            contador += 1
        cola.put(numero)
    
    return contador


# Nota: la igualdad de diccionarios funciona bien (sin importar el orden)