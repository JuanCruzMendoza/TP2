module Solucion where

-- Ejercicio 1
relacionesValidas :: [(String, String)] -> Bool
relacionesValidas [] = True
relacionesValidas ((x,y):xs) | x == y = False
                             | not (repetidos (x,y) xs) = relacionesValidas xs
                             | otherwise = False

repetidos :: (String, String) -> [(String, String)] -> Bool
repetidos _ [] = False
repetidos (x,y) ((w,z):xs) | (x,y) == (w,z) || (y,x) == (w,z) = True
                           | otherwise = repetidos (x,y) xs


-- Ejercicio 2
personas :: [(String, String)] -> [String]
personas xs = personas_ xs []

personas_ :: [(String,String)] -> [String] -> [String]
personas_ [] _ = []
personas_ ((x,y):xs) l | not (pertenece x l) && not (pertenece y l) = x:y:personas_ xs (x:y:l)
                       | not (pertenece x l) = x:personas_ xs (x:l)
                       | not (pertenece y l) = y:personas_ xs (y:l)

pertenece :: String -> [String] -> Bool
pertenece _ [] = False
pertenece x (y:xs) | x == y = True
                   | otherwise = pertenece x xs


-- Ejercicio 3
amigosDe :: String -> [(String,String)] -> [String]
amigosDe _ [] = []
amigosDe persona ((x,y):xs) | persona == x = y:amigosDe persona xs
                            | persona == y = x:amigosDe persona xs
                            | otherwise = amigosDe persona xs


-- Ejercicio 4
longitud :: [String] -> Int
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

personaConMasAmigos_aux :: [String] -> [(String,String)] -> String
personaConMasAmigos_aux [x] _ = x
personaConMasAmigos_aux (x:y:xs) relaciones | longitud (amigosDe x relaciones) > longitud (amigosDe y relaciones) = personaConMasAmigos_aux (x:xs) relaciones
                                            | otherwise = personaConMasAmigos_aux (y:xs) relaciones

personaConMasAmigos :: [(String,String)] -> String
personaConMasAmigos relaciones = personaConMasAmigos_aux (personas relaciones) relaciones


--Resolucion profe ej1
relacionesValidas2 :: [(String,String)] -> Bool
relacionesValidas2 [] = True
relacionesValidas2 (x:xs) = (fst x /= snd x) && not (elem x xs) && not (elem (snd x, fst x) xs) && relacionesValidas2 xs