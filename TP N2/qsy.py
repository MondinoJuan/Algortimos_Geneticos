from itertools import combinations


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

mochilas_bus = busqueda_exhaustiva(elementos1, 4200) 
print(mochilas_bus[0])

print("Ejercicio 1")
mochila1_bus = busqueda_exhaustiva(elementos1, 4200)
print("Precio máximo (Exhaustiva):", mochila1_bus[0][-1], "; Volumen: ", mochila1_bus[0][-2])
print("Elementos en mochila (Exhaustiva):", [e[0] for e in mochila1_bus[0][:-2]])
