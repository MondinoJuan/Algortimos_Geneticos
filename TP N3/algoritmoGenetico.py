import random
import os
import matplotlib.pyplot as plt
import pandas as pd

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

def calculadorAptitudes(poblacion, distancias):
    aptitudes = []
    for individuo in poblacion:
        distanciaTotal = 0
        for i in range(len(individuo) - 1):
            capitalActual = individuo[i]
            siguienteCapital = individuo[i + 1]
            distanciaTotal += distancias[capitalActual, siguienteCapital]
        distanciaTotal += distancias[individuo[-1], individuo[0]]
        fitness = 1 / distanciaTotal if distanciaTotal > 0 else 0
        aptitudes.append(fitness)
    return aptitudes

def calculadorEstadisticos(poblacion, aptitudes):
    maxAptitudes = max(aptitudes)
    minAptitudes = min(aptitudes)
    mejorCromosoma = poblacion[aptitudes.index(maxAptitudes)]
    avgAptitudes = round((sum(aptitudes)/len(aptitudes)),4)
    return [maxAptitudes, minAptitudes, avgAptitudes, mejorCromosoma]

def seleccionRuleta(poblacion, aptitudes, cantidad):
    sumaAptitudes = sum(aptitudes)
    if sumaAptitudes == 0:
        return random.choices(poblacion, k=cantidad)
    
    aptitudesNormalizadas = [apt / sumaAptitudes for apt in aptitudes]
    acumuladas = []
    acum = 0
    for fitness in aptitudes:
        acum += fitness
        acumuladas.append(acum)
    seleccionados = []
    for _ in range(cantidad):
        probAleatoria = random.random()
        for i, acum in enumerate(acumuladas):
            if probAleatoria <= acum:
                seleccionados.append(poblacion[i][:])
                break
    return seleccionados

def seleccionTorneo(poblacion, aptitudes, cantidadIndividuos, cantidadCompetidores):
    ganadores = []
    for _ in range(cantidadIndividuos):
        indices = random.sample(range(len(poblacion)), cantidadCompetidores)
        competidores = [poblacion[i] for i in indices]
        aptitudesCompetidores = [aptitudes[i] for i in indices]
        indiceGanador = aptitudesCompetidores.index(max(aptitudesCompetidores))
        ganadores.append(competidores[indiceGanador][:])
    return ganadores

def crossoverCiclico(padre1, padre2):
    n = len(padre1)
    hijo1 = [-1] * n
    hijo2 = [-1] * n
    ciclos = []
    visitado = [False] * n
    for i in range(n):
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

def mutacionInvertida(poblacion, probMutacion):
    for i in range(len(poblacion)):
        if random.random() < probMutacion:
            individuo = poblacion[i]
            pos1 = random.randint(0, len(individuo) - 1)
            pos2 = random.randint(0, len(individuo) - 1)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            segmento_invertido = individuo[pos1:pos2+1][::-1]
            poblacion[i] = individuo[:pos1] + segmento_invertido + individuo[pos2+1:]
    return poblacion

def ciclosSinElitismo(distancias, cantidadCiclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadCompetidores):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    poblacion = generarPoblacion(cantidadIndividuos,cantidadGenes)
    aptitudes = calculadorAptitudes(poblacion, distancias)
    estadisticos = calculadorEstadisticos(poblacion, aptitudes)
    maximos.append(estadisticos[0])
    minimos.append(estadisticos[1])
    promedios.append(estadisticos[2])
    mejores.append(estadisticos[3])
    for _ in range (cantidadCiclos):
        if opcionSeleccion == "1":
            poblacion = seleccionRuleta(poblacion, aptitudes, cantidadIndividuos)
        else:
            poblacion = seleccionTorneo(poblacion, aptitudes, cantidadIndividuos, cantidadCompetidores) 
        for i in range (0,len(poblacion),2):
            if i + 1 < len(poblacion):
                padre1 = poblacion[i]
                padre2 = poblacion[i+1]
            if random.random() < probCrossover :
                hijo1, hijo2 = crossoverCiclico(padre1,padre2)
                poblacion[i], poblacion[i+1] = hijo1, hijo2
        poblacion = mutacionInvertida(poblacion, probMutacion)
        aptitudes = calculadorAptitudes(poblacion, distancias)
        estadisticos = calculadorEstadisticos(poblacion, aptitudes)
        maximos.append(estadisticos[0])
        minimos.append(estadisticos[1])
        promedios.append(estadisticos[2])
        mejores.append(estadisticos[3])
    return maximos, minimos, promedios, mejores

def ciclosConElitismo(distancias, ciclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, metodoSeleccion, cantidadElitismo, cantidadCompetidores = None):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    poblacion = generarPoblacion(cantidadIndividuos,cantidadGenes)
    aptitudes = calculadorAptitudes(poblacion, distancias)
    estadisticos = calculadorEstadisticos(poblacion, aptitudes)
    maximos.append(estadisticos[0])
    minimos.append(estadisticos[1])
    promedios.append(estadisticos[2])
    mejores.append(estadisticos[3])
    for _ in range (ciclos):
        elitistas = [] 
        aptitudesOrdenadas = sorted(aptitudes, reverse = True)
        for i in range(cantidadElitismo):
            indice = aptitudes.index(aptitudesOrdenadas[i])
            elitistas.append(pob[indice])
        if metodoSeleccion == 1:
            pobIntermedia = seleccionRuleta(pob, aptitudes, cantidadIndividuos - cantidadElitismo)
        else:
            pobIntermedia = seleccionTorneo(pob, aptitudes, cantidadIndividuos - cantidadElitismo, cantidadCompetidores) 
        for i in range (0,len(pobIntermedia),2):
            padre = pobIntermedia[i]
            madre = pobIntermedia[i+1]
            if random.random() < probCrossover :
                hijo1, hijo2 = crossoverCiclico(padre,madre)
                pobIntermedia[i], pobIntermedia[i+1] = hijo1, hijo2
        pobIntermedia = mutacionInvertida(pobIntermedia, probMutacion)
        pob = pobIntermedia + elitistas
        poblacion = generarPoblacion(cantidadIndividuos,cantidadGenes)
        aptitudes = calculadorAptitudes(poblacion, distancias)
        estadisticos = calculadorEstadisticos(poblacion, aptitudes)
        maximos.append(estadisticos[0])
        minimos.append(estadisticos[1])
        promedios.append(estadisticos[2])
        mejores.append(estadisticos[3])
        
    return maximos, minimos, promedios, mejores

def binarioADecimal(cromosoma):
    decimal = 0
    exponente=0
    for i in range(len(cromosoma)-1,-1,-1):
        if cromosoma[i] == 1:
            decimal = decimal + pow(2,exponente)
        exponente += 1 
    return decimal

def crearTabla(maximos, minimos, promedios, mejores, metodoSeleccion, opcionElitismo):
    nombreMetodo = '_Ruleta' if metodoSeleccion == 1 else '_Torneo'
    nombreElitismo = '_Elitismo' if opcionElitismo == 2 else ''
    nombreCantidadCiclos = str(len(maximos) - 1)
    cadenas = ['->'.join(str(ciudad) for ciudad in cromosoma) for cromosoma in mejores]
    distancias_totales = [f"{1/max_apt:.2f}" if max_apt > 0 else "inf" for max_apt in maximos]
    dfNuevo = pd.DataFrame({
        'Generacion': range(len(maximos)),
        'Max_Fitness': maximos,
        'Min_Fitness': minimos,
        'AVG_Fitness': promedios,
        'Distancia_km': distancias_totales,
        'Mejor_Recorrido': cadenas,
    })
    
    archivoExcel = f'AG_{nombreCantidadCiclos}Ciclos{nombreMetodo}{nombreElitismo}.xlsx'
    if os.path.exists(archivoExcel):
        os.remove(archivoExcel)
        dfNuevo.to_excel(archivoExcel, index=False)
    else:
        dfNuevo.to_excel(archivoExcel, index=False)

def generarGrafico(maximos, minimos, promedios, mejores, titulo, ciclo):
    x = list(range(len(maximos)))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, maximos, label='Máximos', marker='o', linestyle='-', color='b', linewidth=1.5, markersize=3)
    ax.plot(x, minimos, label='Mínimos', marker='o', linestyle='-', color='g', linewidth=1.5, markersize=3)
    ax.plot(x, promedios, label='Promedios', marker='o', linestyle='-', color='r', linewidth=1.5, markersize=3)
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xlabel('GENERACIÓN', fontsize=12)
    ax.set_ylabel('APTITUD (Fitness)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    nombre_archivo = titulo.replace(" ", "_") + '.png'
    plt.savefig(nombre_archivo, dpi=300)
    print(f"Gráfico guardado en: {nombre_archivo}")
    plt.show()

def algoritmoGenetico(opcionElitismo, opcionSeleccion, distancias):
    cantidadCiclos = 200
    probCrossover = 0.75
    probMutacion = 0.05
    cantidadIndividuos = 50
    cantidadGenes = 24 #! Está mal el enunciado? Con 23 no tiene en cuenta CABA
    cantidadElitismo = 2
    cantidadCompetidores = int(cantidadIndividuos * 0.4)
    maximosPorCiclo = []
    minimosPorCiclo = []
    promediosPorCiclo = []
    if opcionElitismo == 1:
        maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclosSinElitismo(distancias, cantidadCiclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadCompetidores)
        if opcionSeleccion == 1:
            titulo = 'Seleccion RULETA - de '+ str(cantidadCiclos) + ' ciclos'
        else:
            titulo = 'Seleccion TORNEO - de '+ str(cantidadCiclos) + ' ciclos'
        generarGrafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, cantidadCiclos)
        crearTabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, opcionSeleccion, opcionElitismo)
    else:
        maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclosConElitismo(distancias, cantidadCiclos, probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, opcionSeleccion, cantidadElitismo, cantidadCompetidores)
        if opcionSeleccion == 1:
            titulo = 'Seleccion RULETA ELITISTA - de '+ str(cantidadCiclos) + ' ciclos'
        else:
            titulo = 'Seleccion TORNEO ELITISTA - de '+ str(cantidadCiclos) + ' ciclos'
        generarGrafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, cantidadCiclos)
        crearTabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, opcionSeleccion, opcionElitismo)
        print(f"\n{'='*60}")
    print(f"RESULTADO FINAL:")
    print(f"Mejor distancia encontrada: {1/maximosPorCiclo[-1]:.2f} km")
    print(f"Mejor recorrido: {mejores[-1]}")
    #print(f"Tiempo de ejecución: {demora:.2f} segundos")
    print(f"{'='*60}\n")