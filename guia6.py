import math
# Ej 1
def imprimir_hola():
    print("hola mundo")
    
def imprimir_un_verso():
    print("Que ves \nque ves")
    
def raizDe2() -> float:
    res:float = round(2**(1/2),4)
    return res

def factorial_de_dos() -> int:
    res:int = 2
    return res

def perimetro() -> float:
    res:float = math.pi * 2
    return res

# Ej 2
def imprimir_un_saludo(nombre:str):
    print("Hola" + nombre)

def raiz_cuadrada_de(numero:float) -> float:
    res:float = numero**(1/2)
    return res

def fahrenheit_a_celsius(t:float) -> float: 
    res:float = ((t-32)*5)/9
    return res

def imprimir_estribillo(estribillo:str)->str:
    res:str = estribillo*2
    return res

def es_multiplo_de(n:int,m:int)->bool:
    return (n % m)== 0

def es_par(numero:int)->bool:
    return es_multiplo_de(numero,2)

def cantidad_de_pizzas(comensales:int, min_cant_de_porciones:int)->int:
    return (comensales * min_cant_de_porciones) // 8
    

# Ej 3
def alguno_es_0(numero1:float, numero2:float):
    return numero1 == 0 or numero2 == 0

def ambos_son_0(numero1:float, numero2:float):
    return numero1 == 0 and numero2 == 0

def es_nombre_largo(nombre:str)->bool:
    return len(str) >= 3 and len(str) <=8

def es_bisiesto(año:str):
    return (año % 400 == 0) or (año % 4 == 0 and not(año % 100 ==0))


# Ej 4
def peso_pino(altura:int)->int:
    if altura <= 3:
        return altura * 100 * 3
    else:
        return 3*300*3 + (altura - 3)*100*2
    
def es_peso_util(peso:int)->bool:
    return 400 <= peso <= 1000

def sirve_pino(altura:int)->bool:
    return es_peso_util(peso_pino(altura))


# Ej 5
def devolver_el_doble_si_es_par(numero:int)->int:
    if numero % 2 == 0:
        return numero*2
    else:
        return numero
    
def devolver_valor_si_es_par_sino_el_que_sigue(numero:int)->int:
    if numero % 2 == 0:
        return numero
    else:
        return numero +1
    
def doble_o_triple(numero:int)->int:
    if numero%9 ==0:
        return 3*numero
    elif numero %3 == 0:
        return 2*numero
    else:
        return numero
    
def lindo_nombre(nombre:str):
    if len(nombre) >= 5:
        print("Tu nombre tiene muchas letras")
    else:
        print("Tu nombre tiene menos de 5 caracteres")
        
def elRango(numero:int):
    if numero < 5:
        print("menor a 5")
    elif 10<=numero<=20:
        print("entre 10 y 20")
    elif numero >20:
        print("mayor que 20")
        
def vacaciones_trabajar(edad:int, sexo:str):
    if (edad < 18) or (edad >= 60 and sexo == "F") or (edad >= 65 and sexo== "M"):
        print("Vacaciones")
    else:
        print("Trabajar")
        
        
# Ej 6
def uno_diez():
    i = 0
    while i < 10:
        i +=1
        print(i)

def pares_10_40():
    i = 10
    while i <= 40:
        print(i)
        i+=2

def eco():
    print("eco"*10)
    
def cuenta_regresiva(num:int):
    while num >=1:
        print(num)
        num -=1
        
    print("despegue")
    
def viaje_en_el_tiempo(partida:int,llegada:int):
    while partida != llegada:
        partida -= 1
        print("Viajo un año en el pasado, estamos en el año " + str(partida))
        
def viaje_aristoteles(partida:int):
    partida -= 20
    while partida > -384:
        print("Viajo 20 años al pasado,e stamos en el año " + str(partida))
        partida -= 20


# Ej 7
def uno_diez_for():
    for i in range(10):
        print(i+1)

def cuenta_regresiva_for(num:int):
    for i in range(num,0,-1):
        print(i)
        
    print("despegue")
    
def pares_10_40_for():
    for i in range(10,40+2,2):
        print(i)
        
# Ej 8
def rt(x: int, g: int) -> int:
    g = g + 1
    return x + g
g: int = 0

def ro(x: int) -> int:
    global g
    g = g + 1
    return x + g