def busquedaHeuristica(distancias, indexCapital):
    n = len(distancias)
    visitadas = [False] * n
    visitadas[indexCapital] = True
    recorrido = [indexCapital]
    distanciasParciales = []
    distanciaTotal = 0
    indexCiudadActual = indexCapital
    for _ in range(n - 1):
        mejorDistancia = float('inf')
        indexMejorCiudad = -1
        for indexCiudad in range(n):
            if not visitadas[indexCiudad]:
                distancia = distancias[indexCiudadActual, indexCiudad]
                if distancia < mejorDistancia:
                    mejorDistancia = distancia
                    indexMejorCiudad = indexCiudad
        distanciasParciales.append(mejorDistancia)
        distanciaTotal += mejorDistancia
        recorrido.append(indexMejorCiudad)
        visitadas[indexMejorCiudad] = True
        indexCiudadActual = indexMejorCiudad
    distanciaVuelta = distancias[indexCiudadActual, indexCapital]
    distanciasParciales.append(distanciaVuelta)
    distanciaTotal += distanciaVuelta
    recorrido.append(indexCapital)
    return recorrido, distanciasParciales, distanciaTotal
