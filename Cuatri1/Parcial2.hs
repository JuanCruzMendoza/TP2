-- Parcial 28/11/2022

{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}

--1
pertenece :: (Eq t) => t -> [t] -> Bool
pertenece n [] = False
pertenece n (x:xs) = n == x || pertenece n xs

enPosicionesPares :: Integer -> [Integer] -> Bool
enPosicionesPares n [] = True
enPosicionesPares n [x] = n == x
enPosicionesPares n (x:xs) = n == x && enPosicionesPares n (tail xs)

esElUnico :: Integer -> [Integer] -> Bool
esElUnico n (x:xs) = xs == [] && n == x

-- 2
--a)
-- requiere: true
--asegura: s es estrictamente creciente hasta su valor max y ahi empieza a decrecer,
-- s tiene al menos 3 elementos, sin repetidos, el max no puede ser el primero ni el ultimo
subeBaja :: [Integer] -> Bool
subeBaja l | longitud l < 3 = False
           | elemRepetidos l = False
           | maximo l == head l || maximo l == ultimo l = False
           | otherwise = subeBaja_aux l (maximo l)

subeBaja_aux :: [Integer] -> Integer -> Bool
subeBaja_aux (x:xs) elemMax | x == elemMax = baja (x:xs)
                            | x < head xs = subeBaja_aux xs elemMax
                            | otherwise = False

baja :: [Integer] -> Bool
baja [x] = True
baja (x:xs) = x > head xs && baja xs

longitud :: (Eq t) => [t] -> Integer
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

ultimo :: [t] -> t
ultimo [x] = x
ultimo (x:xs) = ultimo xs

elemRepetidos :: (Eq t) => [t] -> Bool
elemRepetidos [] = False
elemRepetidos (x:xs) = pertenece x xs || elemRepetidos xs

maximo :: (Ord t) => [t] -> t
maximo [x] = x
maximo (x:xs) | x >= head xs = maximo (x:tail xs)
              | otherwise = maximo xs

--b)
cambiarElem :: [Integer] -> [Integer] -> [[Integer]]
cambiarElem [] elem = []
cambiarElem (x:xs) elem | not(pertenece x elem) = (elem ++ [x]):cambiarElem xs elem
                       | otherwise = cambiarElem xs elem

combinaciones_aux :: [[Integer]] -> [Integer] -> [[Integer]]
combinaciones_aux [] seq = []
combinaciones_aux (x:xs) seq = cambiarElem seq x  ++ combinaciones_aux (cambiarElem seq x) seq ++ combinaciones_aux xs seq

combinaciones :: [Integer] -> [[Integer]]
combinaciones seq = combinaciones_aux (cambiarElem seq []) seq

combinacionesSubeBaja_aux :: [[Integer]] -> [[Integer]]
combinacionesSubeBaja_aux [] = []
combinacionesSubeBaja_aux (x:xs) | subeBaja x = x:combinacionesSubeBaja_aux xs
                                 | otherwise = combinacionesSubeBaja_aux xs

combinacionesSubeBaja :: [Integer] -> [[Integer]]
combinacionesSubeBaja seq = combinacionesSubeBaja_aux (combinaciones seq)