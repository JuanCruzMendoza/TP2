-- Ejercicio 1
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

-- Ejercicio 2
parteEntera :: Float -> Int
parteEntera n | n >= 0 && n < 1 = 0
              | n < 0 && n > -1 = 0
              | n >= 1 = 1 + parteEntera (n-1)
              | n <= -1 = -1 + parteEntera (n+1)

-- Ejercicio 3
esDivisible :: Int -> Int -> Bool
esDivisible x y | x > 0 = esDivisible (x-y) y
                | otherwise = x == 0

-- Ejercicio 4
sumaImpares :: Int -> Int
sumaImpares 1 = 1
sumaImpares n = 2*n-1 + sumaImpares (n-1)