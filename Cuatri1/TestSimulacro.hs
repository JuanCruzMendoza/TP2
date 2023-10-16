module TestSimulacro where

import Test.HUnit
import Solucion2

run = runTestTT tests

tests = test [
    "relacionesValidas: lista vacía" ~: relacionesValidas [] ~?= True,
    "relacionesValidas: 1 relacion" ~: relacionesValidas [("Carlos","Juan")] ~?= True,
    "relacionesValidas: 2 relaciones" ~: relacionesValidas [("Carlos","Juan"),("Tanya", "Pablo")] ~?= True,
    "relacionesValidas: invalida nombre repetido" ~: relacionesValidas [("Carlos","Carlos")] ~?= False,
    "relacionesValidas: valida repetido" ~: relacionesValidas [("Carlos","Juan"), ("Carlos","Tanya")] ~?= True,
    "relacionesValidas: invalida relacion repetida" ~: relacionesValidas [("Carlos","Juan"), ("Juan","Carlos")] ~?= False,

    "personas: lista vacía" ~: personas [] ~?= [],
    "personas: 1 relación" ~: iguales (personas [("Carlos", "Juan")]) ["Carlos", "Juan"] ~?= True,
    "personas: 2 relaciones" ~: iguales (personas [("Carlos","Juan"),("Tanya", "Pablo")]) ["Carlos", "Juan", "Tanya","Pablo"] ~?= True,
    "personas: 2 relaciones repetidos" ~: iguales (personas [("Carlos","Juan"), ("Carlos","Tanya")]) ["Carlos", "Juan", "Tanya"] ~?= True,

    "amigosDe: lista vacía" ~: amigosDe "Carlos" [] ~?= [],
    "amigosDe: sin relaciones" ~: amigosDe "Carlos" [("Tanya", "Pablo")] ~?= [],
    "amigosDe: 1 amigo" ~: amigosDe "Carlos" [("Carlos","Juan"),("Tanya", "Pablo")] ~?= ["Juan"],
    "amigosDe: 2 amigos" ~: iguales (amigosDe "Carlos" [("Carlos","Juan"),("Tanya", "Pablo"),("Tanya","Carlos")]) ["Juan","Tanya"] ~?= True,

    "personaConMasAmigos: 1 relación" ~: contenida [personaConMasAmigos [("Carlos","Juan")]] ["Carlos","Juan"] ~?= True,
    "personaConMasAmigos: 2 relaciones, 1 max" ~: personaConMasAmigos [("Carlos","Juan"),("Tanya", "Carlos")] ~?= "Carlos"
    ]

contenida :: (Eq t) => [t] -> [t] -> Bool
contenida [] _ = True
contenida (x:xs) ys = pertenece x ys && contenida xs ys

iguales :: (Eq t) => [t] -> [t] -> Bool
iguales xs ys = contenida xs ys && contenida ys xs
