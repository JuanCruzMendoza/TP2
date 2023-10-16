module Ejercicios_parciales where
-- Ejercicios de repaso pre parcial

-- 1 Encontrar el índice de un elemento de una lista
-- requiere que el elemento esté en la lista
indice :: (Eq t) => [t] -> t -> Integer
indice [] _ = 0
indice (x:xs) y | x == y = indice [] y
                | otherwise = 1 + indice xs y

-- 2 
eliminarRepetidos :: (Eq t) => [t] -> [t]
eliminarRepetidos [] = []
eliminarRepetidos (x:xs) | pertenece x xs = eliminarRepetidos xs
                         | otherwise = x:eliminarRepetidos xs

pertenece :: (Eq t) => t -> [t] -> Bool
pertenece _ [] = False
pertenece y (x:xs) = x == y || pertenece y xs

-- 3
cantRepetidos :: (Eq t) => t -> [t] -> Integer
cantRepetidos _ [] = 0
cantRepetidos x (y:ys) | x == y = 1 + cantRepetidos x ys
                       | otherwise = cantRepetidos x ys

duplasElementoCant :: (Eq t) => [t] -> [t] -> [(t,Integer)]
duplasElementoCant [] _ = []
duplasElementoCant (x:xs) repetidos | cantRepetidos x repetidos -1 > 0 = (x, cantRepetidos x repetidos -1):duplasElementoCant xs repetidos
                                    | otherwise = duplasElementoCant xs repetidos

eliminarYcontarRepetidos :: (Eq t) => [t] -> ([t],[(t,Integer)])
eliminarYcontarRepetidos repetidos = (eliminarRepetidos repetidos, duplasElementoCant (eliminarRepetidos repetidos) repetidos)


-- Enunciado Parcial

-- 1 y 2
codificar :: [(Char,Char)] -> [Char] -> [Char]
-- requiere: los primeros valores de la tupla son diferentes entre sí, al igual que los segundos,
-- y todos los elementos de c están entre los primeros valores de las tuplas
-- asegura: si un elemento de c es igual al primer valor de una tupla de b, entonces el segundo valor está en res
codificar _ [] = []
codificar codificacion (c:cs) = cambiarChar codificacion c : codificar codificacion cs

cambiarChar :: [(Char,Char)] -> Char -> Char
-- No hay caso base porque c pertenece a la codificacion
cambiarChar ((x,y):xs) c | x == c = y
                         | otherwise = cambiarChar xs c

esCodigo :: [(Char, Char)] -> [[Char]] -> [[Char]] -> Bool
esCodigo cod m n = iguales (aplicoCodigo cod m) n

aplicoCodigo :: [(Char,Char)] -> [[Char]] -> [[Char]]
aplicoCodigo _ [] = []
aplicoCodigo cod (m:ms) = codificar cod m : aplicoCodigo cod ms

-- 3
aprobado :: [Integer] -> Integer
-- requiere notas no vacia y que sean entre 0 y 10
aprobado notas | mayor4 notas && promedio notas >= 7 = 1
               | mayor4 notas && promedio notas >= 4 = 2
               | otherwise = 3
               
mayor4 :: [Integer] -> Bool
mayor4 [] = True
mayor4 (x:xs) = x >= 4 && mayor4 xs

promedio :: [Integer] -> Float
promedio notas = fromInteger (sumaLista notas) / fromInteger (longitud notas)

sumaLista :: (Num t) => [t] -> t
sumaLista [] = 0
sumaLista (x:xs) = x + sumaLista xs

longitud :: [t] -> Integer
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

-- Auxiliares
contenido :: (Eq t) => [t] -> [t] -> Bool
contenido [] _ = True
contenido (x:xs) ys = pertenece x ys && contenido xs ys

iguales :: (Eq t) => [t] -> [t] -> Bool
iguales xs ys = contenido xs ys && contenido ys xs



