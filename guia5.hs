-- Ejercicio 1
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use even" #-}
{-# HLINT ignore "Use list literal pattern" #-}
longitud :: [t] -> Int
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

ultimo :: [t] -> t
ultimo (x:[]) = x
ultimo (x:xs) = ultimo xs

reverso :: [t] -> [t]
reverso (x:[]) = [x]
reverso (x:xs) = (reverso xs) ++ [x]

principio :: [t] -> [t]
principio (x:xs) | longitud (x:xs) == 1 = []
                 | otherwise = x : principio xs

-- Ejercicio 2
pertenece :: (Eq t) => t -> [t] -> Bool
pertenece n [] = False
pertenece n (x:xs) | n == x = True
                   | null xs = False
                   | otherwise = pertenece n xs

todosIguales :: (Eq t) => [t] -> Bool
todosIguales (x:[]) = True
todosIguales (x:xs) = x == head xs && todosIguales xs

todosDistintos :: (Eq t) => [t] -> Bool
todosDistintos [] = True -- Definimos un caso límite
todosDistintos (x:xs) | xs == [] = True
                      | pertenece x xs = False
                      | otherwise = todosDistintos xs

hayRepetidos :: (Eq t) => [t] -> Bool
hayRepetidos s = not (todosDistintos s)

quitar :: (Eq t) => t -> [t] -> [t]
quitar n [] = []
quitar n (x:xs) | n == x = xs
                | otherwise = [x] ++ quitar n xs

quitarTodos :: (Eq t) => t -> [t] -> [t]
quitarTodos n [] = []
quitarTodos n xs | pertenece n xs = quitarTodos n (quitar n xs)
                 | otherwise = xs

eliminarRepetidos :: (Eq t) => [t] -> [t]
eliminarRepetidos [] = []
eliminarRepetidos (x:xs) | pertenece x xs = [x] ++ eliminarRepetidos (quitarTodos x xs) 
                         | otherwise = [x] ++ eliminarRepetidos xs 

mismosElementos :: (Eq t) => [t] -> [t] -> Bool
mismosElementos [] [] = True
mismosElementos (x:xs) l | pertenece x l = mismosElementos (quitarTodos x xs) (quitarTodos x l)
                         | otherwise = False

valorPosicion :: (Num t, Eq t) => t -> [a] -> a
valorPosicion i (x:xs) | i == 0 = x 
                       | otherwise = valorPosicion (i-1) xs

capicua :: (Eq t) => [t] -> Bool
capicua seq | longitud seq == 1 || longitud seq == 0 = True 
               | head seq == ultimo (tail seq) = capicua (principio (tail seq))
               | otherwise = False

-- Ejercicio 3
sumatoria :: [Int] -> Int
sumatoria [] = 0
sumatoria (x:xs) = x + sumatoria xs

productoria :: [Int] -> Int
productoria [] = 1
productoria (x:xs) = x * productoria xs

maximo :: [Int] -> Int
maximo (x:[]) = x
maximo (x:xs) | x > head xs = maximo (x:tail xs)
              | otherwise = maximo xs

sumarN :: Int -> [Int] -> [Int]
sumarN _ [] = []
sumarN n (x:xs) = (n+x):sumarN n xs

sumarElPrimero :: [Int] -> [Int]
sumarElPrimero xs = sumarN (head xs) xs

sumarElUltimo :: [Int] -> [Int]
sumarElUltimo xs = sumarN (ultimo xs) xs

pares :: [Int] -> [Int]
pares [] = []
pares (x:xs) | mod x 2 == 0 = x : pares xs
             | otherwise = pares xs

multiplosDeN :: Int -> [Int] -> [Int]
multiplosDeN _ [] = []
multiplosDeN n (x:xs) | mod x n == 0 = x: multiplosDeN n xs
                      | otherwise = multiplosDeN n xs

ordenar :: [Int] -> [Int]
ordenar [] = []
ordenar xs =  ordenar (quitar (maximo xs) xs) ++ [maximo xs]


quicksort :: [Int] -> [Int]
quicksort [] = []
quicksort (x:xs) = menores x xs ++ [x] ++ mayores x xs

mayores :: Int -> [Int] -> [Int]
mayores _ [] = []
mayores n (x:xs) | n < x = x:mayores n xs
                 | otherwise = mayores n xs

menores :: Int -> [Int] -> [Int]
menores _ [] = []
menores n (x:xs) | n > x = x:menores n xs
                 | otherwise = menores n xs           


-- Ejercicio 4
sacarBlancosRepetidos :: [Char] -> [Char]
sacarBlancosRepetidos (x:[]) = []
sacarBlancosRepetidos (x:xs) | x == ' ' && head xs == ' ' = sacarBlancosRepetidos xs 
                             | otherwise = x:sacarBlancosRepetidos xs 