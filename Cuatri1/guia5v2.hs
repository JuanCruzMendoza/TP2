reverso :: [t] -> [t]
reverso (x:[]) = [x]
reverso (x:xs) = (reverso xs) ++ [x]

capicua :: (Eq t) => [t] -> Bool
capicua [] = True
capicua xs = xs == reverso xs

sumatoria :: [Integer] -> Integer
sumatoria [] = 0
sumatoria (x:xs) = x + sumatoria xs

productoria :: [Integer] -> Integer
productoria [] = 0
productoria [x] = x
productoria (x:xs) = x * productoria xs

maximo :: [Integer] -> Integer 
maximo [x] = x
maximo (x:xs) | x > head xs = (maximo (x:tail xs))
              | otherwise = maximo xs

sumarN :: Integer -> [Integer] -> [Integer]
sumarN n [] = []
sumarN n (x:xs) = (x+n) : sumarN n xs

sumarElPrimero :: [Integer] -> [Integer]
sumarElPrimero xs = sumarN (head xs) xs

ultimo :: [t] -> t
ultimo [x] = x
ultimo (x:xs) = ultimo xs

sumarElUltimo :: [Integer] -> [Integer]
sumarElUltimo xs = sumarN (ultimo xs) xs

pares :: [Integer] -> [Integer]
pares [] = []
pares (x:xs) | mod x 2 == 0 = x : pares xs
             | otherwise = pares xs

multiplosDeN :: Integer -> [Integer] -> [Integer]
multiplosDeN _ [] = [] 
multiplosDeN n (x:xs) | mod x n == 0 = x: multiplosDeN n xs
                      | otherwise = multiplosDeN n xs

quitar :: (Eq t) => t -> [t] -> [t]
quitar n [] = []
quitar n (x:xs) | n == x = xs
                | otherwise = x: quitar n xs

ordenar :: [Integer] -> [Integer]
ordenar [x] = [x]
ordenar xs = ordenar (quitar (maximo xs) xs) ++ [maximo xs]

sacarBlancosRepetidos :: [Char] -> [Char]
sacarBlancosRepetidos [] = []
sacarBlancosRepetidos [x] = [x] 
sacarBlancosRepetidos (x:y:xs) | x == ' ' && y == ' ' = sacarBlancosRepetidos (y:xs)
                               | otherwise = x:sacarBlancosRepetidos (y:xs)

contarPalabras :: [Char] -> Integer
contarPalabras [] = 1
contarPalabras (x:xs) | x == ' ' = 1 + contarPalabras xs
                      | otherwise = contarPalabras xs

palabras :: [Char] -> [Char] -> [[Char]]
palabras l [] = [l]
palabras l (x:xs) | x == ' ' = l:palabras [] xs
                  | otherwise = palabras (l++[x]) xs

palabras_ :: [Char] -> [[Char]]
palabras_ xs = palabras [] xs

longitud :: [t] -> Int
longitud [] = 0
longitud (x:xs) = 1 + longitud xs

palabraMasLarga2 :: [[Char]] -> [Char]
palabraMasLarga2 [x] = x
palabraMasLarga2 (x:y:xs) | longitud x > longitud y = palabraMasLarga2 (x:xs)
                         | otherwise = palabraMasLarga2 (y:xs)

palabraMasLarga2_ :: [Char] -> [Char]
palabraMasLarga2_ xs = palabraMasLarga2 (palabras_ xs)

aplanar :: [[Char]] -> [Char]
aplanar [x] = x
aplanar (x:xs) = x++[' ']++aplanar xs

sumaAcumulada :: (Num t) => [t] -> t -> [t]
sumaAcumulada [] _ = []
sumaAcumulada (x:xs) l = (x+l): sumaAcumulada xs (x+l)

sumaAcumulada' :: (Num t) => [t] -> [t]
sumaAcumulada' xs = sumaAcumulada xs 0

primos :: Integer -> Integer -> [Integer]
primos 1 _ = []
primos n i | mod n i == 0 = i:primos (div n i) i
           | otherwise = primos n (i+1)

descomponerEnPrimos :: [Integer] -> [[Integer]] 
descomponerEnPrimos [] = []
descomponerEnPrimos (x:xs) = primos x 2 : descomponerEnPrimos xs

sumaAcumulada2 :: [Integer] -> [Integer]
sumaAcumulada2 [x] = [x]
sumaAcumulada2 (x:xs) = x: sumaAcumulada2 ((x+head xs): tail xs)