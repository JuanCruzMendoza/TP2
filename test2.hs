todosMenores :: (Int, Int, Int) ->Bool
todosMenores (x,y,z) = f(x)>g(x) && f(y)>g(y) && f(z)>g(z)

g :: Int -> Int
g n | mod n 2 == 0 = div n 2
    | otherwise = 3*n+1

f :: Int->Int
f n | n<=7 = n*n
    | otherwise = 2*n-1

absFloat :: Float -> Float
absFloat n | n>=0 = n 
     | otherwise = (-n)

distanciaManhattan:: (Float, Float, Float) ->(Float, Float, Float) ->Float
distanciaManhattan (x,y,z) (h,i,j) = absFloat x-h+y-i+z-j