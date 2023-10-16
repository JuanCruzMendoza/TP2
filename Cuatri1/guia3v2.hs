
absoluto :: Float -> Float
absoluto x | x < 0 = -x
           | otherwise = x

maximoAbsoluto :: Float -> Float -> Float
maximoAbsoluto x y | absoluto x > absoluto y = x
                   | otherwise = y

maximo3 :: Float -> Float -> Float -> Float
maximo3 x y z | x >= y && x >= z = x
              | y >= x && y >= z = y
              | otherwise = z

algunoEs0 :: Float -> Float -> Bool
algunoEs0 0 _ = True
algunoEs0 _ 0 = True
algunoEs0 _ _ = False

algunoEs02 :: Float -> Float -> Bool
algunoEs02 x y = x == 0 || y == 0

ambosSon0 :: Float -> Float -> Bool
ambosSon0 0 0 = True
ambosSon0 _ _ = False

ambosSon02 :: Float -> Float -> Bool
ambosSon02 x y = x == 0 && y == 0

mismoIntervalo :: Float -> Float -> Bool
mismoIntervalo x y | x <= 3 = y <= 3
                   | x > 7 = y > 7
                   | otherwise = y > 3 && y <= 7

digitoDecenas :: Integer -> Integer
digitoDecenas n = mod (div n 10) 10

todoMenor :: (Integer, Integer) -> (Integer, Integer) -> Bool
todoMenor (x,y) (w,z) = x < w && y < z


sumarMultiplo :: Integer -> Integer -> Integer
sumarMultiplo x n | mod x n == 0 = x
                  | otherwise = 0

sumarSoloMultiplos :: (Integer,Integer,Integer) -> Integer -> Integer
sumarSoloMultiplos (x,y,z) n = sumarMultiplo x n + sumarMultiplo y n + sumarMultiplo z n

bisiesto :: Integer -> Bool
bisiesto n = not (mod n 4 /= 0 || (mod n 100 == 0 && mod n 400 /= 0))