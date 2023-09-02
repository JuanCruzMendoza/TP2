absoluto :: Int -> Int
absoluto n | n>=0 = n 
     | otherwise = -n

maximoabsoluto :: Int -> Int -> Int
maximoabsoluto x y | absoluto x >= absoluto y = x
    | absoluto y > absoluto x = y

g1 n | n==1 = 130
    | n==2 = 120

f1 n | n==130 = 30
    | n==120 = 20

h :: Int -> Int
h n = f1 (g1 n)

maximo3 :: (Int, Int, Int) -> Int
maximo3 (x,y,z) | x > y && x>z = x
                | y>x && y>z = y
                | otherwise = z

func n = (if n==1 then 2 else 4)

-- Con guardas
algunoEs0 :: Float -> Float -> Bool
algunoEs0 x y | x == 0 || y==0 = True | otherwise = False

-- Con pattern matching
algunoEs0_p :: Float -> Float -> Bool
algunoEs0_p 0 y = True
algunoEs0_p x 0 = True
algunoEs0_p _ _ = False

ambosSon0 :: Float -> Float -> Bool
ambosSon0 x y | x == 0 && y == 0 = True
 | otherwise = False

ambosSon0_p :: Float -> Float -> Bool
ambosSon0_p 0 0 = True
ambosSon0_p _ _ = False

intervalo :: Float -> Int
intervalo x | (x <= 3 ) = 0 | (x > 3 && x <= 7) = 1 | otherwise = 2

mismoIntervalo :: Float -> Float -> Bool
mismoIntervalo x y | intervalo x == intervalo y = True
 | otherwise = False

sumarDinstintos :: Int -> Int -> Int -> Int
sumarDinstintos x y z | x == z = y
 | x == z = y
 | x == y = z
 | y == z = x
 | y == z && x == z = 0
 | otherwise = x+y+z


multiplos :: Int -> Int -> Bool
multiplos x y | x<0 && y<0 = error "negativos"
 | mod x y == 0 = True
 | otherwise = False

digitosUnidades :: Int -> Int
digitosUnidades n | n<0 = error "negativos"
 | otherwise = mod n 10 

digitosDecenas :: Int -> Int
digitosDecenas n | n<0 = error "negativos"
 | otherwise = mod (div n 10) 10

estanRelacionados :: Int -> Int -> Bool
estanRelacionados x y | x==0 || y==0 = error "Distinto de cero"
                      | mod x y == 0 = True
                      | otherwise = False


prodInt :: (Float, Float) -> (Float, Float) -> Float
prodInt x y = fst x * fst y + snd x * snd y

todoMenor :: (Float, Float) -> (Float, Float) -> Bool
todoMenor x y = fst x < fst y && snd x < snd y

distanciaPuntos  :: (Float, Float) -> (Float, Float) -> Float
distanciaPuntos x y = sqrt ((fst x - fst y)**2 + (snd x - snd y)**2)

sumaTerna :: (Int, Int, Int)-> Int
sumaTerna (x,y,z) = x+y+z

multiplosSuma :: Int -> Int -> Int
multiplosSuma x y | x<0 && y<0 = error "negativos"
 | mod x y == 0 = x
 | otherwise = 0

sumarSoloMultiplos :: (Int, Int, Int) -> Int -> Int
sumarSoloMultiplos (x,y,z) w | w <0 = error "Negativo"
                             | otherwise = multiplosSuma x w + multiplosSuma y w + multiplosSuma z w

posPrimerPar :: (Int, Int, Int) -> Int
posPrimerPar (x,y,z) | mod x 2 == 0 = 0
                     | mod y 2 == 0 = 1
                     | mod z 2 == 0 = 2
                     | otherwise = 4
                
crearPar :: a -> b -> (a,b)
crearPar a b = (a,b)

invertir :: (a,b) -> (b,a)
invertir (a,b) = (b,a)

bisiesto :: Int -> Bool
bisiesto x = (not (mod x 100 == 0) || mod x 400 == 0) && mod x 4 == 0 

sumarLista a | a == [] = 0
             | otherwise = head a + sumarLista (tail a)

todosMenores :: (Int, Int, Int) ->Bool
todosMenores (x,y,z) = f(x)>g(x) && f(y)>g(y) && f(z)>g(z)

g :: Int -> Int
g n | mod n 2 == 0 = div n 2
    | otherwise = 3*n+1

f :: Int->Int
f n | n<=7 = n*n
    | otherwise = 2*n-1

distanciaManhattan:: (Float, Float, Float) -> (Float, Float, Float) -> Float
distanciaManhattan (x,y,z) (h,i,j) = abs (x-h+y-i+z-j)

sumaUltimosDosDigitos :: Int -> Int
sumaUltimosDosDigitos n = digitosDecenas n + digitosUnidades n

comparar :: Int -> Int -> Int
comparar x y | sumaUltimosDosDigitos x < sumaUltimosDosDigitos y = 1
             | sumaUltimosDosDigitos x > sumaUltimosDosDigitos y = -1
             | sumaUltimosDosDigitos x == sumaUltimosDosDigitos y = 0
