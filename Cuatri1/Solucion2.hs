{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}
{-# HLINT ignore "Use foldr" #-}
module Solucion2 where
--Ejercicio 1
relacionesValidas :: [(String, String)] -> Bool
relacionesValidas [] = True
relacionesValidas ((x,y):xs) | x == y = False
                             | pertenece (x,y) xs || pertenece (y,x) xs = False
                             | otherwise = relacionesValidas xs

pertenece :: (Eq t) => t -> [t] -> Bool
pertenece _ [] = False
pertenece x (y:xs) = x == y || pertenece x xs 

-- Ejercicio 2
personas :: [(String, String)] -> [String]
-- Requiere relacionesValidas(relaciones)
personas relaciones = eliminarRepetidos (personasRepetidas relaciones)

personasRepetidas :: [(String, String)] -> [String]
personasRepetidas [] = []
personasRepetidas ((x,y):xs) = x:y:personasRepetidas xs

eliminarRepetidos :: [String] -> [String]
eliminarRepetidos [] = []
eliminarRepetidos (x:xs) | pertenece x xs = eliminarRepetidos xs
                         | otherwise = x:eliminarRepetidos xs

-- Ejercicio 3
amigosDe :: String -> [(String, String)] -> [String]
-- requiere relacionesValidas(relaciones)
amigosDe _ [] = []
amigosDe persona ((x,y):xs) | persona == x = y:recursion
                            | persona == y = x:recursion
                            | otherwise = recursion
                            where recursion = amigosDe persona xs

-- Ejercicio 4
longitud :: [t] -> Int
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

personaConMasAmigos_aux :: [String] -> [(String,String)] -> String
--requiere lista no vacía, y relacionesValidas(relaciones)
personaConMasAmigos_aux [p] _ = p
personaConMasAmigos_aux (p1:p2:ps) relaciones | longitud( amigosDe p1 relaciones) >= longitud( amigosDe p2 relaciones) = personaConMasAmigos_aux (p1:ps) relaciones
                                              | otherwise = personaConMasAmigos_aux (p2:ps) relaciones

personaConMasAmigos :: [(String, String)] -> String
personaConMasAmigos relaciones = personaConMasAmigos_aux (personas relaciones) relaciones