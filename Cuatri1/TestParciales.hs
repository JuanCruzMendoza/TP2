module TestParciales where

import Ejercicios_parciales
import Test.HUnit

run = runTestTT tests

tests = test [
    "eliminarRepetidos: sin repetidos" ~: eliminarRepetidos [1,2,3] ~?= [1,2,3],
    "eliminarRepetidos: varios repetidos" ~: iguales (eliminarRepetidos [1,1,5,2,1,4,2,3]) [1,5,2,4,3] ~?= True,

    "eliminarYcontarRepetidos: sin repetidos" ~: eliminarYcontarRepetidos [1,2] ~?= ([1,2],[]),
    "eliminarYcontarRepetidos: 1 repetido" ~: iguales (fst (eliminarYcontarRepetidos [1,2,1])) [1,2] && 
    iguales (snd (eliminarYcontarRepetidos [1,2,1])) [(1,1)] ~?= True,
    "eliminarYcontarRepetidos: varios repetidos" ~: iguales (fst (eliminarYcontarRepetidos [1,2,2,1,3,4,2,3])) [1,2,3,4] && 
    iguales (snd (eliminarYcontarRepetidos [1,2,2,1,3,4,2,3])) [(1,1),(2,2),(3,1)] ~?= True,

    "codificar: sin codigo" ~: codificar [('a','A'),('b','B'),('c','C')] [] ~?= [],
    "codificar: letras repetidas" ~: codificar [('a','A'),('b','B'),('c','C')] "abbcaca" ~?= "ABBCACA",

    "esCodigo: vacio m" ~: esCodigo [('a','A'),('b','B'),('c','C')] [] ["ABC"] ~?= False,
    "esCodigo: vacio n" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc"] [] ~?= False,
    "esCodigo: 1 codigo valido" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc"] ["ABC"] ~?= True,
    "esCodigo: 1 codigo valido" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc"] ["ABC"] ~?= True,
    "esCodigo: codigos validos" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc","aacbbabc"] ["ABC","AACBBABC"] ~?= True,
    "esCodigo: n mayor invalido" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc"] ["ABC","AACBBABC"] ~?= False,
    "esCodigo: m mayor invalido" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc","aacbbabc"] ["AACBBABC"] ~?= False,
    "esCodigo: distinto invalido" ~: esCodigo [('a','A'),('b','B'),('c','C')] ["abc","caacbbabc"] ["ABC","AACBBABC"] ~?= False
    ]

