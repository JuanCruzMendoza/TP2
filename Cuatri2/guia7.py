# EJercicio 1

def pertenece1(s: list[int], e:int)-> bool:
    for i in s:
        if i == e:
            return True
        else:
            continue
    return False

def pertenece2(s:list[int], e:int)->bool:
    return e in s

def pertenece3(s:list[int],e:int)->bool:
    i:int = 0
    while 0<=i<len(s):
        if s[i] == e:
            return True
        else:
            i+=1
    return False

def pertenece(s,e)->bool:
  for i in s:
    if i == e:
      return True
  return False

def divideATodos(s:list[int],e:int)->bool:
  for i in s:
    if i % e == 0:
      continue
    else:
      return False
  return True

def sumaTotal(s:list[int])->int:
  suma: int= 0
  for i in s:
    suma+=i
  return suma

def ordenados(s:list[int])->bool:
  i:int = 0
  while 0<=i <(len(s)-1):
    if s[i] >= s[i+1]:
      return False
    i+=1
  return True

def long7(s:list[str])->bool:
  for i in s:
    if len(i)>7:
      return True
  return False

def palindromo(s:str)->bool:
  return s == s[::-1]

def password(s:str)->str:
  def minus(s:str)->bool:
    for i in s:
      if i >= "a" and i<= "z":
        return True
    return False

  def mayus(s:str)->bool:
    for i in s:
      if i>="A" and i<="Z":
        return True
    return False

  def digito(s:str)->bool:
    for i in s:
      if i>="0" and i<= "9":
        return True
    return False

  if len(s) > 8 and minus(s) and mayus(s) and digito(s):
    return "VERDE"

  elif len(s) < 5:
    return "ROJA"
  else:
    return "AMARILLA"

def banco(s:list[tuple])->int:
  saldo:int = 0
  for i,j in s:
    if i == "I":
      saldo+=j
    elif i == "R":
      saldo-=j
  return saldo

def vocales(s:str)->bool:
  voc:list[str] = ["a","e","i","o","u"]
  for i in s:
    if i.lower() in voc:
      voc.remove(i.lower())
  return len(voc) <= 2


# Ejercicio 2
def pares_cero_inout(s:list[int])->list[int]:
  for i in range(1,len(s),2):
    s[i] = 0
  return s

def pares_cero_in(s:list[int])->list[int]:
  output: list[int] = []
  for i in range(1,len(s)+1):
    if i % 2 == 0:
      output.append(0)
    else:
      output.append(s[i])
  return output

def sin_vocales(s:str)->str:
  voc:str = "aeiou"
  output:str = ""
  for i in s:
    if i.lower() in voc:
      continue
    else:
      output += i
  return output

def remplaza_vocales(s:str)->str:
  voc:str = "aeiou"
  res:str = ""
  for i in s:
    if i in voc:
      res+= "-"
    else:
      res+= i
  return res

def daVueltaStr(s:str)->str:
  output:str = ""
  for i in range(len(s)-1,-1,-1):
    output += s[i]
  return output

def eliminarRepetidos(s:str)->str:
  output:str=""
  for i in s:
    if not pertenece(output, i):
      output+=i
  return output


# Ejercicio 3
def aprobado(notas:list[int])->int:
  def mayores4(l:list[int])->bool:
    for i in l:
      if i < 4:
        return False
    return True

  promedio:float = sumaTotal(notas) / len(notas)

  if mayores4(notas) and promedio >= 7:
    return 1
  elif mayores4(notas) and 4<=promedio<7:
    return 2
  else:
    return 3


# Ejercicio 4
def estudiantes()->list[str]:
  estudiantes:list[str] = []
  nombre = input()
  while nombre != "listo":
    estudiantes.append(nombre)
    nombre = input()
  return estudiantes

def sube()->list[tuple]:
  historial: list[tuple] = []
  operacion:str=""
  saldo = 0
  while operacion != "X":

    operacion = input("Operación:")
    if operacion == "C":
      saldo_cargar = int(input("Saldo a cargar:"))
      saldo += saldo_cargar
      historial.append(("C",saldo_cargar))

    elif operacion == "D":
      saldo_descontar = int(input("Saldo a descontar:"))
      saldo -= saldo_descontar
      historial.append(("D",saldo_descontar))

  return historial, saldo

import random
def sietemedio():
  cant_jugadores:int = int(input("Cantidad de jugadores: "))
  cartas:list = [1,2,3,4,5,6,7,10,11,12]
  historial:list = []

  for i in range(cant_jugadores):
    mano:list = []
    contador:float = 0
    ganador = {}
    print("Jugador "+str(i+1))
    primera:float = random.choice(cartas)
    print("Primera carta: "+str(primera))
    mano.append(primera)

    if primera <= 7:
      contador += primera
    else:
      contador += 0.5


    while contador < 7.5:
      decision = input("plantarse o seguir:")

      if decision == "plantarse":
        break
      elif decision == "seguir":
        nueva_carta: float = random.choice(cartas)
        mano.append(nueva_carta)
        print("Nueva carta: "+str(nueva_carta))
        if nueva_carta <= 7:
          contador += nueva_carta
        else:
          contador += 0.5

    if contador > 7.5:
      print("Perdiste")
    else:
      print("Puntaje: "+str(contador))
      ganador[contador] = i+1

    historial.append((mano,contador))

  if ganador == {}:
    print("Todos perdieron")
  else:
    puntaje_ganador = max(ganador.keys())
    for i in ganador.keys():
      if i == puntaje_ganador:
        print("Ha ganado jugador " +str(ganador[puntaje_ganador]))

  return historial

def perteneceACadaUno(seq:list[list[int]], e:int)->list[bool]:
  out:list[bool] = []
  for i in seq:
    if e in i:
      out.append(True)
    else:
      out.append(False)
  return out

def esMatriz(s:list[list[int]])->bool:
  if len(s) > 0 and len(s[0]) > 0:
    for i in range(len(s)):
      if len(s[i]) != len(s[0]):
        return False
    return True
  return False

def filasOrdenadas(s:list[list[int]])->bool:
  out: list[bool] = []
  for i in s:
    if ordenados(i):
      out.append(True)
    else:
      out.append(False)
  return out

import numpy as np
def matriz_potencia(d:int,p:int)->list[list[float]]:
  def copia(s:list[list[float]])->list[list[float]]:
    out:list[list[float]] = []
    for i in range(len(s)):
      out.append([])
      for j in range(len(s[0])):
        out[i].append(s[i][j])
    return out

  m = [[1,2],[3,2]]
  out:list[list[float]]= copia(m)
  for _ in range(p-1):
    out_actual:list[list[float]] = copia(out)
    for i in range(len(m)):
      for j in range(len(m[0])):
        res:float=0
        for k in range(d):
          res+= out[i][k]*m[k][j]
        out_actual[i][j]= res
    out = copia(out_actual)
  return out


