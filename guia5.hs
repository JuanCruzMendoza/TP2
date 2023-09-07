pertenece :: (Eq t) => t -> [t] -> Bool
pertenece n (x:xs) | n == x = True
                   | null xs = False
                   | otherwise = pertenece n xs