from time import perf_counter

def busquedaHeuristica(distancias, indiceCapitalInicial, opcion=None):
    if opcion == 1:
        tiempoInicial = perf_counter()
    cantCapitales = len(distancias)
    visitadas = [False] * cantCapitales
    visitadas[indiceCapitalInicial] = True
    recorrido = [indiceCapitalInicial]
    distanciasParciales = []
    distanciaTotal = 0
    indiceCapitalActual = indiceCapitalInicial
    for _ in range(cantCapitales - 1):
        mejorDistancia = float('inf')
        indiceMejorCapital = -1
        for indiceCapital in range(cantCapitales):
            if not visitadas[indiceCapital]:
                distancia = distancias[indiceCapitalActual, indiceCapital]
                if distancia < mejorDistancia:
                    mejorDistancia = distancia
                    indiceMejorCapital = indiceCapital
        distanciasParciales.append(mejorDistancia)
        distanciaTotal += mejorDistancia
        recorrido.append(indiceMejorCapital)
        visitadas[indiceMejorCapital] = True
        indiceCapitalActual = indiceMejorCapital
    distanciaVuelta = distancias[indiceCapitalActual, indiceCapitalInicial]
    distanciasParciales.append(distanciaVuelta)
    distanciaTotal += distanciaVuelta
    recorrido.append(indiceCapitalInicial)
    if opcion == 1:
        tiempoFinal = perf_counter()
        demora = tiempoFinal - tiempoInicial
        return recorrido, distanciasParciales, distanciaTotal, demora
    else:
        return recorrido, distanciasParciales, distanciaTotal

def mejorRecorrido(distancias):
    tiempoInicial = perf_counter()
    cantCapitales = len(distancias)
    distanciasTotales = []
    for indiceCapital in range(cantCapitales):
        _, _, distanciaTotal = busquedaHeuristica(distancias, indiceCapital)
        distanciasTotales.append(float(distanciaTotal))
    indiceMejorDistancia = distanciasTotales.index(min(distanciasTotales))
    recorrido, distanciasParciales, distanciaTotal = busquedaHeuristica(distancias, indiceMejorDistancia)
    tiempoFinal = perf_counter()
    demora = tiempoFinal - tiempoInicial
    return recorrido, distanciasParciales, distanciaTotal, indiceMejorDistancia, demora