absoluto :: Int -> Int
absoluto n | n>=0 = n 
     | otherwise = -n

maximoabsoluto :: Int -> Int -> Int
maximoabsoluto x y | absoluto x >= absoluto y = x
    | absoluto y > absoluto x = y

g n | n==1 = 130
    | n==2 = 120

f n | n==130 = 30
    | n==120 = 20

h :: Int -> Int
h n = f (g n)

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
 | otherwise = mod n 100
 