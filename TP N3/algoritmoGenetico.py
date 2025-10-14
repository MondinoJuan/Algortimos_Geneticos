import random
import os
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter

CAPITALES = ["CABA", "Córdoba", "Corrientes", "Formosa", "La Plata", "La Rioja", "Mendoza", "Neuquén", "Paraná", "Posadas", "Rawson", "Resistencia", 
            "Río Gallegos", "San Fernando del Valle de Catamarca", "San Miguel de Tucumán", "San Salvador de Jujuy", "Salta", "San Juan", "San Luis", 
            "Santa Fe", "Santa Rosa", "Santiago del Estero", "Ushuaia", "Viedma"]

def generarPermutacion(cantidadGenes):
    cromosoma = list(range(cantidadGenes))
    random.shuffle(cromosoma)
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes):
    poblacion = []
    for _ in range(cantidadCromosomas):
        cromosoma = generarPermutacion(cantidadGenes)
        poblacion.append(cromosoma)
    return poblacion

def funcionObjetivo(x):
    return 1 / x

def calculadorObjetivo(poblacion, distancias):
    objetivos = []
    for individuo in poblacion:
        distanciaTotal = 0
        for i in range(len(individuo) - 1):
            capitalActual = individuo[i]
            siguienteCapital = individuo[i + 1]
            distanciaTotal += distancias[capitalActual][siguienteCapital]
        distanciaTotal += distancias[individuo[-1]][individuo[0]]
        objetivo = funcionObjetivo(distanciaTotal)
        objetivos.append(objetivo)
    return objetivos

def calculadorAptitudes(poblacion, distancias):
    aptitudes = []
    objetivos = calculadorObjetivo(poblacion, distancias)
    sumObj = sum(objetivos)
    for objetivo in objetivos:
        fitness = objetivo / sumObj if sumObj > 0 else 0 
        aptitudes.append(fitness)
    return aptitudes

def calculadorEstadisticos(poblacion, aptitudes):
    maxAptitudes = max(aptitudes)
    minAptitudes = min(aptitudes)
    mejorCromosoma = poblacion[aptitudes.index(maxAptitudes)][:]
    avgAptitudes = (sum(aptitudes)/len(aptitudes))
    return [maxAptitudes, minAptitudes, avgAptitudes, mejorCromosoma]

def seleccionRuleta(poblacion, aptitudes, cantidadIndividuos):
    fitnessTotal = sum(aptitudes)
    if fitnessTotal == 0:
        return [random.choice(poblacion)[:] for _ in range(cantidadIndividuos)]
    aptitudesNormalizadas = [aptitud / fitnessTotal for aptitud in aptitudes]
    acumuladas = []
    acumulador = 0
    for fitness in aptitudesNormalizadas:
        acumulador += fitness
        acumuladas.append(acumulador)
    seleccionados = []
    for _ in range(cantidadIndividuos):
        probAleatoria = random.random()
        for i, acum in enumerate(acumuladas):
            if probAleatoria <= acum:
                seleccionados.append(poblacion[i][:])
                break
    return seleccionados

def seleccionTorneo(poblacion, aptitudes, cantidadIndividuos, cantidadCompetidores):
    seleccionados = []
    for _ in range(cantidadIndividuos):
        indices = random.sample(range(len(poblacion)), cantidadCompetidores)
        mejor = max(indices, key=lambda i: aptitudes[i])
        seleccionados.append(poblacion[mejor][:])
    return seleccionados

def crossoverCiclico(padre1, padre2):
    cantidadCapitales = len(padre1)
    hijo1 = [-1] * cantidadCapitales
    hijo2 = [-1] * cantidadCapitales
    ciclos = []
    visitado = [False] * cantidadCapitales
    for i in range(cantidadCapitales):
        if not visitado[i]:
            ciclo = []
            pos = i
            while not visitado[pos]:
                visitado[pos] = True
                ciclo.append(pos)
                elemento = padre1[pos]
                pos = padre2.index(elemento)
            ciclos.append(ciclo)
    for i, ciclo in enumerate(ciclos):
        if i % 2 == 0:
            for pos in ciclo:
                hijo1[pos] = padre1[pos]
                hijo2[pos] = padre2[pos]
        else:
            for pos in ciclo:
                hijo1[pos] = padre2[pos]
                hijo2[pos] = padre1[pos]
    return hijo1, hijo2

def mutacionSwap(poblacion, probMutacion):
    poblacionMutada = []
    for individuo in poblacion:
        individuoCopia = individuo[:]
        if random.random() < probMutacion:
            pos1, pos2 = random.sample(range(len(individuoCopia)), 2)
            individuoCopia[pos1], individuoCopia[pos2] = individuoCopia[pos2], individuoCopia[pos1]
        poblacionMutada.append(individuoCopia)
    return poblacionMutada

def ciclosSinElitismo(distancias, cantidadCiclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadCompetidores):
    maximos = []
    minimos = []
    promedios = []
    mejores = []
    poblacion = generarPoblacion(cantidadIndividuos, cantidadGenes)
    aptitudes = calculadorAptitudes(poblacion, distancias)
    estadisticos = calculadorEstadisticos(poblacion, aptitudes)
    maximos.append(estadisticos[0])
    minimos.append(estadisticos[1])
    promedios.append(estadisticos[2])
    mejores.append(estadisticos[3][:])
    for _ in range(cantidadCiclos):
        if opcionSeleccion == 1:
            poblacion = seleccionRuleta(poblacion, aptitudes, cantidadIndividuos)
        else:
            poblacion = seleccionTorneo(poblacion, aptitudes, cantidadIndividuos, cantidadCompetidores)
        nuevaPoblacion = []
        for i in range(0, len(poblacion), 2):
            if i + 1 < len(poblacion):
                if random.random() < probCrossover:
                    hijo1, hijo2 = crossoverCiclico(poblacion[i], poblacion[i + 1])
                    nuevaPoblacion.append(hijo1)
                    nuevaPoblacion.append(hijo2)
                else:
                    nuevaPoblacion.append(poblacion[i][:])
                    nuevaPoblacion.append(poblacion[i + 1][:])
            else:
                nuevaPoblacion.append(poblacion[i][:])
        poblacion = mutacionSwap(nuevaPoblacion, probMutacion)
        aptitudes = calculadorAptitudes(poblacion, distancias)
        estadisticos = calculadorEstadisticos(poblacion, aptitudes)
        maximos.append(estadisticos[0])
        minimos.append(estadisticos[1])
        promedios.append(estadisticos[2])
        mejores.append(estadisticos[3][:])
    return maximos, minimos, promedios, mejores

def ciclosConElitismo(distancias, cantidadCiclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadElitismo, cantidadCompetidores=None):
    maximos = []
    minimos = []
    promedios = []
    mejores = []
    poblacion = generarPoblacion(cantidadIndividuos, cantidadGenes)
    aptitudes = calculadorAptitudes(poblacion, distancias)
    estadisticos = calculadorEstadisticos(poblacion, aptitudes)
    maximos.append(estadisticos[0])
    minimos.append(estadisticos[1])
    promedios.append(estadisticos[2])
    mejores.append(estadisticos[3][:])
    for _ in range(cantidadCiclos):
        indicesOrdenados = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i], reverse=True)
        elitistas = [poblacion[i][:] for i in indicesOrdenados[:cantidadElitismo]]
        if opcionSeleccion == 1:
            pobIntermedia = seleccionRuleta(poblacion, aptitudes, cantidadIndividuos - cantidadElitismo)
        else:
            pobIntermedia = seleccionTorneo(poblacion, aptitudes, cantidadIndividuos - cantidadElitismo, cantidadCompetidores)
        nuevaPoblacion = []
        for i in range(0, len(pobIntermedia), 2):
            if i + 1 < len(pobIntermedia):
                if random.random() < probCrossover:
                    hijo1, hijo2 = crossoverCiclico(pobIntermedia[i], pobIntermedia[i + 1])
                    nuevaPoblacion.append(hijo1)
                    nuevaPoblacion.append(hijo2)
                else:
                    nuevaPoblacion.append(pobIntermedia[i][:])
                    nuevaPoblacion.append(pobIntermedia[i + 1][:])
            else:
                nuevaPoblacion.append(pobIntermedia[i][:])
        pobIntermedia = mutacionSwap(nuevaPoblacion, probMutacion)
        poblacion = pobIntermedia + elitistas
        aptitudes = calculadorAptitudes(poblacion, distancias)
        estadisticos = calculadorEstadisticos(poblacion, aptitudes)
        maximos.append(estadisticos[0])
        minimos.append(estadisticos[1])
        promedios.append(estadisticos[2])
        mejores.append(estadisticos[3][:])
    return maximos, minimos, promedios, mejores

def crearTabla(maximos, minimos, promedios, mejores, metodoSeleccion, opcionElitismo):
    nombreMetodo = '_Ruleta' if metodoSeleccion == 1 else '_Torneo'
    nombreElitismo = '_Elitismo' if opcionElitismo == 2 else ''
    nombreCantidadCiclos = str(len(maximos) - 1)
    cadenas = ['->'.join(CAPITALES[ciudad] for ciudad in cromosoma) for cromosoma in mejores]
    distanciasTotales = [f"{(1/aptitudMaxima - 1):.2f}" if aptitudMaxima > 0 else "inf" for aptitudMaxima in maximos]
    dfNuevo = pd.DataFrame({
        'Generacion': range(len(maximos)),
        'Max_Fitness': maximos,
        'Min_Fitness': minimos,
        'AVG_Fitness': promedios,
        'Distancia_km': distanciasTotales,
        'Mejor_Recorrido': cadenas,
    })
    archivoExcel = f'AG_{nombreCantidadCiclos}Ciclos{nombreMetodo}{nombreElitismo}.xlsx'
    ruta = os.path.join("Excels", archivoExcel)
    if os.path.exists(ruta):
        os.remove(ruta)
    dfNuevo.to_excel(ruta, index=False, float_format="%.8f")
    print(f"Tabla generada: {archivoExcel}")

def generarGrafico(maximos, minimos, promedios, titulo):
    x = list(range(len(maximos)))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, maximos, label='Máximos', color='b', linewidth=1.5)
    ax.plot(x, minimos, label='Mínimos', color='g', linewidth=1.5)
    ax.plot(x, promedios, label='Promedios', color='r', linewidth=1.5)
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xlabel('GENERACIÓN', fontsize=12)
    ax.set_ylabel('APTITUD (Fitness)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    nombreArchivo = titulo.replace(" ", "_") + '.png'
    plt.savefig(f"Fotos/{nombreArchivo}", dpi=300)
    print(f"Gráfico guardado en: {nombreArchivo}")
    plt.close(fig)

def calcularDistanciaRecorrido(recorrido, distancias):
    #! No incluye vuelta al origen
    distanciaTotal = 0
    parciales = []
    for i in range(len(recorrido) - 1):
        distancia = distancias[recorrido[i]][recorrido[i + 1]]
        parciales.append(distancia)
        distanciaTotal += distancia
    distanciaVuelta = distancias[recorrido[-1]][recorrido[0]]
    parciales.append(distanciaVuelta)
    distanciaTotal += distanciaVuelta
    return distanciaTotal, parciales

def algoritmoGenetico(opcionElitismo, opcionSeleccion, distancias):
    tiempoInicial = perf_counter()
    cantidadCiclos = 200000
    probCrossover = 0.85
    probMutacion = 0.10
    cantidadIndividuos = 50
    cantidadGenes = 24
    cantidadElitismo = 2
    cantidadCompetidores = int(cantidadIndividuos * 0.4)
    print(f"\nIniciando Algoritmo Genético...")
    print(f"Parámetros: {cantidadCiclos} ciclos, {cantidadIndividuos} individuos")
    print(f"Probabilidad Crossover: {probCrossover}, Probabilidad Mutación: {probMutacion}\n")
    if opcionElitismo == 1:
        maximos, minimos, promedios, mejores = ciclosSinElitismo(
            distancias, cantidadCiclos, probCrossover, probMutacion,
            cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadCompetidores
        )
        titulo = 'Ruleta' if opcionSeleccion == 1 else 'Torneo'
        titulo = f'Selección {titulo} - {cantidadCiclos} ciclos'
    else:
        maximos, minimos, promedios, mejores = ciclosConElitismo(
            distancias, cantidadCiclos, probCrossover, probMutacion,
            cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadElitismo, cantidadCompetidores
        )
        titulo = 'Ruleta' if opcionSeleccion == 1 else 'Torneo'
        titulo = f'Selección {titulo} ELITISTA - {cantidadCiclos} ciclos'
    generarGrafico(maximos, minimos, promedios, titulo)
    crearTabla(maximos, minimos, promedios, mejores, opcionSeleccion, opcionElitismo)
    mejorRecorrido = mejores[-1][:]
    distanciaTotal, distParciales = calcularDistanciaRecorrido(mejorRecorrido, distancias)
    indiceOrigen = mejorRecorrido[0]
    recorridoCompleto = mejorRecorrido + [indiceOrigen]
    tiempoFinal = perf_counter()
    demora = tiempoFinal - tiempoInicial
    return recorridoCompleto, distParciales, distanciaTotal, demora