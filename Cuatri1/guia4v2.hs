{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}

fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

parteEntera :: Float -> Integer
parteEntera x | x >= 0 && x < 1 = 0
              | x >= 1 = 1 + parteEntera (x-1)
              | x < 1 = -parteEntera (-x)

esDivisible :: Integer -> Integer -> Bool
-- esto sirve para enteros, y no sólo naturales
esDivisible 0 _ = True
esDivisible n m | abs n < abs m = False
                | n < 0 = esDivisible (-n) m
                | m < 0 = esDivisible n (-m)
                | otherwise = esDivisible (n-m) m

absoluto :: Integer -> Integer
absoluto x | x >0 = x
           | otherwise = -x

sumaImpares :: Integer -> Integer
sumaImpares 0 = 0
sumaImpares n = 2*n-1 + sumaImpares (n-1)

medioFact :: Integer -> Integer
--requiere n >=0
medioFact 1 = 1
medioFact 0 = 1
medioFact n = n* medioFact (n-2)

sumaDigitos :: Integer -> Integer
sumaDigitos 0 = 0
sumaDigitos n = mod n 10 + sumaDigitos (div n 10)

todosDigitosIguales :: Integer -> Bool
-- requiere n > 0
todosDigitosIguales n | n < 10 = True
                      | mod n 10 == mod (div n 10) 10 = todosDigitosIguales (div n 10)
                      | otherwise = False

iesimoDigitoReves :: Integer -> Integer -> Integer
-- requiere: n >=0 , 1<= i < cantDigitos(n)
iesimoDigitoReves n 1 = mod n 10
iesimoDigitoReves n i = iesimoDigitoReves (div n 10) (i-1)

iesimoDigito :: Integer -> Integer -> Integer
-- requiere: n >=0 , 1<= i < cantDigitos(n)
iesimoDigito n i = mod (div n (10^((cantDigitos n)-i))) 10

cantDigitos :: Integer -> Integer
cantDigitos n | n < 10 = 1
              | otherwise = 1 + cantDigitos (div n 10)


-- Ejercicio 9
esCapicua :: Integer -> Bool
-- requiere: n >= 0
esCapicua n = esCapicua_aux n 1

esCapicua_aux :: Integer -> Integer -> Bool
esCapicua_aux n i | i > div (cantDigitos n) 2 = True
                  | iesimoDigito n i == iesimoDigito n (cantDigitos n - i +1) = esCapicua_aux n (i+1)
                  | otherwise = False

-- Ej 10 
f1 :: Integer -> Integer
-- n >= 0
f1 0 = 1
f1 n = 2^n + f1 (n-1)

f2 :: Integer -> Float -> Float
-- n >=1,q real
f2 0 _ = 0
f2 1 q = q
f2 n q = q^n + f2 (n-1) q

f3 :: Integer -> Float -> Float
f3 n q = f2 (2*n) q

f4 :: Integer -> Float -> Float
f4 n q = f3 n q - f2 n q


-- Ej 11
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n-1)

eAprox :: Integer -> Float
eAprox 0 = 1 / fromInteger (factorial 0)
eAprox n = 1 / fromInteger (factorial n) + eAprox (n-1)


-- Ej 12
raizDe2Aprox_aux :: Integer -> Float
raizDe2Aprox_aux 1 = 2
raizDe2Aprox_aux n = 2 + 1/ raizDe2Aprox_aux (n-1)

raizDe2Aprox :: Integer -> Float
raizDe2Aprox n = raizDe2Aprox_aux n -1

--Ej 13
suma_potencia_aux :: Integer -> Integer -> Integer
suma_potencia_aux i 1 = i
suma_potencia_aux i j = i^j + suma_potencia_aux i (j-1)

suma_potencia :: Integer -> Integer -> Integer
suma_potencia 1 m = suma_potencia_aux 1 m
suma_potencia n m = suma_potencia_aux n m + suma_potencia (n-1) m

-- Ej 14 
sumaPotencias :: Integer -> Integer -> Integer -> Integer
sumaPotencias 0 _ _ = 0
sumaPotencias n m q = sumaPotencias_aux n m q + sumaPotencias (n-1) m q

sumaPotencias_aux :: Integer -> Integer -> Integer -> Integer
sumaPotencias_aux _ 0 _= 0
sumaPotencias_aux a b q = q^(a+b) + sumaPotencias_aux a (b-1) q

-- Ej 15
sumaRacionales :: Integer -> Integer -> Float
sumaRacionales 0 _ = 0
sumaRacionales p q = sumaRacionales_aux p q + sumaRacionales (p-1) q

sumaRacionales_aux :: Integer -> Integer -> Float
sumaRacionales_aux _ 0 = 0
sumaRacionales_aux p q = (fromIntegral p / fromIntegral q) + sumaRacionales_aux p (q-1)

-- Ej 16
menorDivisor :: Integer -> Integer
menorDivisor n = menorDivisor_aux n 2

menorDivisor_aux :: Integer -> Integer -> Integer
menorDivisor_aux n i | mod n i == 0 = i
                     | otherwise = menorDivisor_aux n (i+1)

esPrimo :: Integer -> Bool
esPrimo n = menorDivisor n == n

sonCoprimos :: Integer -> Integer -> Bool
sonCoprimos n 1 = True
sonCoprimos n m | mod n (menorDivisor m) == 0 = False
                | otherwise = sonCoprimos n (div m (menorDivisor m))

nEsimoPrimo :: Integer -> Integer
nEsimoPrimo n = encuentraPrimos n 3

encuentraPrimos :: Integer -> Integer -> Integer
encuentraPrimos n i | n == 1 = i-1
                    | esPrimo i = encuentraPrimos (n-1) (i+1)
                    | otherwise = encuentraPrimos n (i+1)


--Ej 17
esFibonacci :: Integer -> Bool
esFibonacci n = esFibonacci_aux n 0

esFibonacci_aux :: Integer -> Integer -> Bool
esFibonacci_aux n i | n == fibonacci i = True
                    | n < fibonacci i = False
                    | otherwise = esFibonacci_aux n (i+1)


-- Ej 18
mayorDigitoPar :: Integer -> Integer
-- cambie el require: si n es impar, res=1
mayorDigitoPar n | mod n 2 == 0 = 2*mayorDigitoPar (div n 2)
                 | otherwise = 1

-- Ej 19
esSumaInicialDePrimos_aux :: Integer -> Integer -> Bool
esSumaInicialDePrimos_aux n i | n == 0 = True  
                              | n < 0 = False
                              | esPrimo i = esSumaInicialDePrimos_aux (n-i) (i+1)
                              | otherwise = esSumaInicialDePrimos_aux n (i+1)

esSumaInicialDePrimos :: Integer -> Bool
esSumaInicialDePrimos n = esSumaInicialDePrimos_aux n 2


-- Ej 20
sumarDivisores_aux :: Integer -> Integer -> Integer
sumarDivisores_aux n i | n == i = i
                    | mod n i == 0 = i + sumarDivisores_aux n (i+1)
                    | otherwise = sumarDivisores_aux n (i+1)

sumarDivisores :: Integer -> Integer
sumarDivisores n = sumarDivisores_aux n 1

tomarValorMax :: Integer -> Integer -> Integer
-- requiere : n1 >=1, n1 <= n2
tomarValorMax n1 n2 | n1 == n2 = n1
                    | sumarDivisores n1  >= sumarDivisores n2 = tomarValorMax n1 (n2-1)
                    | sumarDivisores n1  < sumarDivisores n2 = tomarValorMax (n1+1) n2


-- Ej 21
pitagoras_aux :: Integer ->Integer ->Integer ->Integer
pitagoras_aux n (-1) _ = 0
pitagoras_aux n m r | n^2 + m^2 <= r^2 = 1 + pitagoras_aux n (m-1) r
                    | otherwise = pitagoras_aux n (m-1) r

pitagoras :: Integer -> Integer -> Integer -> Integer
pitagoras (-1) _ _ = 0
pitagoras n m r = pitagoras_aux n m r + pitagoras (n-1) m r