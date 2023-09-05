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

-- Ejercicio 5
medioFact :: Int -> Int
medioFact n | n == 0  || n == 1 = 1
            | otherwise = n * medioFact (n-2)

-- Ejercicio 6
sumaDigitos :: Int ->Int
sumaDigitos 0 = 0
sumaDigitos n = mod n 10 + sumaDigitos (div n 10)

-- Ejercicio 7
todosDigitosIguales :: Int -> Bool
todosDigitosIguales n | n < 10 = True
                      | otherwise = mod n 10 == mod (div n 10) 10 && todosDigitosIguales (div n 10)

-- Ejercico 8 
iesimoDigito :: Int -> Int -> Int
iesimoDigito n i = mod (div n (10^(cantDigitos n - i))) 10

cantDigitos :: Int -> Int
cantDigitos x | x == 0 = 0
              | otherwise = 1 + cantDigitos (div x 10)
--Ejercicio 9
primerDigito :: Int -> Int -> Int
primerDigito n q = iesimoDigito n (cantDigitos n - q)

ultimoDigito :: Int -> Int -> Int
ultimoDigito n q = iesimoDigito n (q+1)

esCapicua_aux :: Int -> Int -> Bool
esCapicua_aux n q | q == div (cantDigitos n) 2 || q == (div (cantDigitos n) 2) +1  = True
              | primerDigito n q == ultimoDigito n q = esCapicua_aux n (q+1)
              | otherwise = False

esCapicua :: Int -> Bool
esCapicua n = esCapicua_aux n 1

-- Ejercicio 10
f1 :: Int -> Int
f1 0 = 1
f1 n = 2^n + f1 (n-1)

f2 :: Int -> Int -> Int
f2 1 q = q
f2 n q = q^n + f2 (n-1) q

f3 :: Int -> Int -> Int
f3 n q = f2 (2*n) q

f4 :: Int -> Int -> Int
f4 n q = f3 n q - f2 n q

-- EJercicio 11
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n-1)

eAprox :: Int -> Float
eAprox 0 = 1
eAprox n = 1 / fromIntegral (factorial n) + eAprox (n-1)

e = eAprox 10

-- Ejercicio 12
sucesion :: Int -> Double
sucesion 1 = 2
sucesion n = 2 + 1 / sucesion (n-1)

raizDe2Aprox :: Int -> Double
raizDe2Aprox n = sucesion n - 1

-- Ejercicio 13
suma_1 :: Int -> Int -> Int
suma_1 _ 0 = 0 
suma_1 n j = n^j + suma_1 n (j-1)

suma_2 :: Int -> Int -> Int
suma_2 0 _ = 0
suma_2 n m = suma_1 n m + suma_2 (n-1) m

-- Ejercicio 14
sumaPotencias_b :: Int -> Int -> Int -> Int
sumaPotencias_b q a 0 = q^a
sumaPotencias_b q a m = q^(a+m) + sumaPotencias_b q a (m-1)

sumaPotencias :: Int -> Int -> Int -> Int
sumaPotencias q 0 m = sumaPotencias_b q 0 m
sumaPotencias q n m = sumaPotencias_b q n m + sumaPotencias q (n-1) m 

-- Ejercicio 15
sumaRacionales_interna :: Int -> Int -> Float
sumaRacionales_interna p 1 = fromIntegral p
sumaRacionales_interna p q = (fromIntegral p) / (fromIntegral q) + sumaRacionales_interna p (q-1)

sumaRacionales :: Int -> Int -> Float
sumaRacionales 1 m = sumaRacionales_interna 1 m
sumaRacionales n m = sumaRacionales_interna n m + sumaRacionales (n-1) m
