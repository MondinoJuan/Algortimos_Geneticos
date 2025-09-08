from itertools import combinations
from math import factorial as f

#elemento = [numero, volumen, precio]

elementos1 = [
    [1, 150, 20],
    [2, 3325, 40],
    [3, 600, 50],
    [4, 805, 36],
    [5, 430, 25],
    [6, 1200, 64],
    [7, 770, 54],
    [8, 60, 18],
    [9, 930, 46],
    [10, 353, 28]
]

#elemento = [num, peso, precio]
elementos2 = [
    [1, 1800, 72],
    [2, 600, 36],
    [3, 1200, 60],
]

def busqueda_exhaustiva(elementos, capacidad_max):
    mochilas = []
    for r in range(1, len(elementos)+1):
        for mochila in combinations(elementos, r):
            sum_precio_mochila = 0
            sum_capacidad_mochila = 0
            for e in mochila:
                sum_capacidad_mochila += e[1]
                sum_precio_mochila += e[2]
            if sum_capacidad_mochila <= capacidad_max:
                mochila = list(mochila)
                mochila.append(sum_capacidad_mochila)
                mochila.append(sum_precio_mochila)
                mochilas.append(mochila)
    mochilas.sort(key=lambda x: x[-1], reverse=True)
    return mochilas

def algoritmo_greedy(elementos:list[list], capacidad_max):
    mochila = []
    elementos_greedy = []
    for e in elementos:
        nombre, capacidad, valor = e
        prop = e[2]/e[1]
        elementos_greedy.append([nombre, capacidad, valor, prop])
    elementos_greedy.sort(key=lambda x: x[-1], reverse=True)
    sum_precio_mochila = 0
    sum_capacidad_mochila = 0
    for e in elementos_greedy:
        if (capacidad_max - e[1]) >= 0:
            capacidad_max -= e[1]
            sum_precio_mochila += e[2]
            sum_capacidad_mochila += e[1]
            mochila.append(e)
    return mochila, sum_precio_mochila, sum_capacidad_mochila

print("Ejercicio 1")
mochila1_bus = busqueda_exhaustiva(elementos1, 4200)
print("Precio máximo (Exhaustiva):", mochila1_bus[0][-1], "; Volumen: ", mochila1_bus[0][-2])
print("Elementos en mochila (Exhaustiva):", [e[0] for e in mochila1_bus[0][:-2]])

print("----------------------------------------------------------------------")

print("Ejercicio 2")
mochila1_greedy, precio1, vol = algoritmo_greedy(elementos1, 4200)
print("Precio obtenido (Greedy):", precio1, "; Volumen: ", vol)
print("Elementos en mochila (Greedy):", [e[0] for e in mochila1_greedy])

print("----------------------------------------------------------------------")

print("Ejercicio 3")
mochila2_bus = busqueda_exhaustiva(elementos2, 3000)
mochila2_greedy, precio2, peso = algoritmo_greedy(elementos2, 3000)
print("Precio máximo (Exhaustiva):", mochila2_bus[0][-1], "; Peso: ", mochila2_bus[0][-2])
print("Precio obtenido (Greedy):", precio2, "; Peso: ", peso)
print("Elementos en mochila (Exhaustiva):", [e[0] for e in mochila2_bus[0][:-2]])
print("Elementos en mochila (Greedy):", [e[0] for e in mochila2_greedy])

