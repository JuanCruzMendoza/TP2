{-# OPTIONS_GHC -Wno-overlapping-patterns #-}
type Usuario = (Integer, String)
-- (id, nombre)

type Relacion = (Usuario, Usuario)

type Publicacion = (Usuario, String, [Usuario])
-- Usuario, texto publicado, likes de usuarios

type RedSocial = ([Usuario],[Relacion],[Publicacion])

usuarios :: RedSocial -> [Usuario]
usuarios (x,y,z) = x

relaciones :: RedSocial -> [Relacion]
relaciones (x,y,z) = y

publicaciones :: RedSocial -> [Publicacion]
publicaciones (x,y,z) = z

idDeUsuario :: Usuario -> Integer
idDeUsuario (x,y) = x

nombreDelUsuario :: Usuario -> String
nombreDelUsuario (x,y) = y

usuarioDePublicacion :: Publicacion -> Usuario
usuarioDePublicacion (x,y,z) = x

textoPublicacion :: Publicacion -> String
textoPublicacion (x,y,z) = y

likesDePublicacion :: Publicacion -> [Usuario]
likesDePublicacion (x,y,z) = z


-- 1
nombresDeUsuarios :: RedSocial -> [String]
-- requiere: redSocialValida(red)
nombresDeUsuarios ([],y,z) = []
nombresDeUsuarios ((x:xs),y,z) = nombreDelUsuario x : nombresDeUsuarios (xs,y,z)

--2
proyectarNombres :: [Usuario] -> [String]
proyectarNombres [(ids, [])] = []
proyectarNombres ((id,nombre):usuarios) | sinRepetidos   nombre : proyectarNombres usuarios

--Auxiliares
pertenece :: (Eq t) => t -> [t] -> Bool
pertenece _ [] = False
pertenece x (y:ys) = x == y || pertenece x ys

mismosElementos :: (Eq t) => [t] -> [t] -> Bool
mismosElementos xs ys = contenido xs ys && contenido ys xs

contenido :: (Eq t) => [t] -> [t] -> Bool
contenido [] _ = True
contenido (x:xs) ys = pertenece x ys && contenido xs ys

usuariosValidos :: [Usuario] -> Bool
usuariosValidos usuarios = usuariosValidos_aux usuarios && noHayIdsRepetidos usuarios

usuariosValidos_aux :: [Usuario] -> Bool
usuariosValidos_aux (usuario:us) = usuarioValido usuario && usuariosValidos_aux us 

usuarioValido :: Usuario -> Bool
usuarioValido (id,nombre) = id > 0 && nombre /= []

noHayIdsRepetidos :: [Usuario] -> Bool
noHayIdsRepetidos [] = True
noHayIdsRepetidos ((id,nombre):usuarios) = not (pertenece id (idsUsuarios usuarios)) && noHayIdsRepetidos usuarios

idsUsuarios :: [Usuario] -> [Integer]
idsUsuarios ((id, nombre):usuarios) = id : idsUsuarios usuarios 

