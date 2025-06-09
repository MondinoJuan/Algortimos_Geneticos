import random
import os

# POBLACION INICIAL

def limpiar_pantalla():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux o Mac
        os.system('clear')

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)]
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes):
    poblacion = []
    for i in range(cantidadCromosomas):
        cromosoma = completoCromosoma(cantidadGenes)
        poblacion.append(cromosoma)
    return poblacion

# FITNESS

def binarioADecimal(cromosoma, cantGenes):
    decimal = 0
    for i in range(cantGenes):
        decimal += cromosoma[i] * (2 ** (cantGenes - 1 - i))
    return decimal

def funcionObjetivo(x, coef):
    return (x / coef) ** 2  

# CROSSOVER

def crossover1Punto(cantGenes, padres):
    lugarCorte = random.randint(1, cantGenes - 1)
    hijo1 = []
    hijo2 = []
    for i in range(lugarCorte):
        hijo1.append(padres[0][i])
        hijo2.append(padres[1][i])
    for i in range(lugarCorte, cantGenes):
        hijo1.append(padres[1][i])
        hijo2.append(padres[0][i])
    hijos = [hijo1, hijo2]
    return hijos

# MUTACION

def mutacionInvertida(cantGenes, hijo):
    if cantGenes < 2:
        return hijo  # No se puede invertir si hay menos de 2 genes
    posInicial = random.randint(0, cantGenes - 2)
    posFinal = random.randint(posInicial + 1, cantGenes - 1)
    segmento = hijo[posInicial:posFinal + 1]
    hijo[posInicial:posFinal + 1] = segmento[::-1]  # Invertir el segmento
    return hijo

# SELECCION

# Ruleta
def seleccionRuleta(poblacion, fitnessValores):
    probAcum = 0
    probAcumulativas = []
    for fitness in fitnessValores:
        probAcum += fitness
        probAcumulativas.append(probAcum)
    padres = []
    indicesPadres = []
    while len(padres) < 2:
        probAleatoria = random.random()
        for i, probAcum in enumerate(probAcumulativas):
            if probAleatoria <= probAcum:
                if i not in indicesPadres:
                    padres.append(poblacion[i])
                    indicesPadres.append(i)
                break
    return padres, indicesPadres

# Torneo
def seleccionTorneo(poblacion):
    pass

# Elitismo
def seleccionElitismo(poblacion):
    pass

#Menu Seleccion

def eleccionSeleccion():
    print("Seleccione el método de selección:")
    print("1. Ruleta")
    print("2. Torneo")
    print("3. Elitismo")
    while True:
        try:
            opcion = int(input("Ingrese una opción (1, 2 o 3): "))
            if opcion in [1, 2, 3]:
                break
            else:
                print("Por favor, ingrese solo 1, 2 o 3.")
        except ValueError:
            print("Por favor, ingrese un número entero válido.")
    limpiar_pantalla()
    return opcion

def menuSeleccion(opc, poblacion, fitnessValores):
    if opc == 1:
        padres, indicesPadres = seleccionRuleta(poblacion, fitnessValores)
    '''
    elif opc == 2:
        padres = seleccionTorneo(poblacion)
    elif opc == 3:
        padres = seleccionElitismo(poblacion)
    '''
    return padres, indicesPadres

# PROGRAMA PRINCIPAL

# Corrida
probCrossover = 0.99 #Cambiar luego a 0.75
probMutacion = 0.99 #Cambiar luego a 0.05
cantidadCromosomas = 10
ciclosPrograma = 1 #Cambiar luego a 20
cantGenes = 5 #Cambiar luego a 30
coef = (2 ** cantGenes) - 1
corridas = 1 #Cambiar luego a 200
maximosPorCiclo = []
minimosPorCiclo = []
promediosPorCiclo = []

opc = 1 #Cambiar luego a eleccionSeleccion()

for corrida in range(corridas):
    print(f"----Corrida Nro {corrida + 1}----")

    poblacion = generarPoblacion(cantidadCromosomas, cantGenes)
    print("\n----Población Inicial----")
    for i, cromosoma in enumerate(poblacion):
        print(f"Cromosoma Inicial Nro {i + 1}: {cromosoma}")
    print()

    mejorCromosoma = None # None y no [] para evitar que se confunda con un cromosoma válido
    mejorFuncObj = -1 # -1 y no 0 para evitar que se confunda con un cromosoma válido

    # Ciclo
    for ciclo in range(ciclosPrograma):
        
        maximosPorCiclo = []
        minimosPorCiclo = []
        promediosPorCiclo = []
        print(f"----Población Nro {ciclo + 1} con sus Aptitudes Físicas (Fitnesses)----")
        funcObjValores = []
        fitnessValores = []
        for crom in poblacion:
            decimal = binarioADecimal(crom, cantGenes)
            funcObjCrom = funcionObjetivo(decimal, coef)
            funcObjValores.append(funcObjCrom)
        suma_funcObj = sum(funcObjValores)
        for crom, funcObjCrom in zip(poblacion, funcObjValores):
            if suma_funcObj == 0:
                fitnessValor = 0
            else:
                fitnessValor = funcObjCrom / suma_funcObj
            fitnessValores.append(fitnessValor)
            print(f"Cromosoma: {crom} → Fitness: {fitnessValor: .6f}")
        print()

        print("----Selección----")
        padres, indicesModificados = menuSeleccion(opc, poblacion, fitnessValores)
        for i, padre in enumerate(padres):
            print(f"Padre Nro {i + 1}: {padre}")
        print()    

        print("----Crossover----")
        if random.uniform(0, 1) <= probCrossover:
            hijos = crossover1Punto(cantGenes, padres)
        else:
            hijos = padres
        for i, hijo in enumerate(hijos):
            print(f"Hijo Nro {i + 1}: {hijo}")
        print()

        print("----Mutaciones----")
        for i in range(len(hijos)):
            if random.uniform(0, 1) <= probMutacion:
                hijos[i] = mutacionInvertida(cantGenes, hijos[i])
        for i,hijo in enumerate(hijos):
            print(f"Mutación en el Hijo Nro {i + 1}: {hijo}")
        print()

        print("----Nueva Población----")
        for i in range(cantidadCromosomas):
            if i in indicesModificados:
                nuevoCrom = hijos[indicesModificados.index(i)]
                poblacion[i] = nuevoCrom
                decimal = binarioADecimal(nuevoCrom, cantGenes)
                funcObjCrom = funcionObjetivo(decimal, coef)
                funcObjValores[i] = funcObjCrom
            print(f"Cromosoma Nuevo Nro {i + 1}: {poblacion[i]}")
        
        maximo = max(funcObjValores)
        minimo = min(funcObjValores)
        promedio = sum(funcObjValores) / cantidadCromosomas
        print(f"Generación {ciclo + 1} → Máx: {maximo:.6f}, Mín: {minimo:.6f}, Prom: {promedio:.6f}\n")

        for crom, funcObjCrom in zip(poblacion, funcObjValores):
            if funcObjCrom > mejorFuncObj:
                mejorCromosoma = crom
                mejorFuncObj = funcObjCrom

    print(f"----Resultados Finales de la Corridad {corrida + 1}----")
    print(f"Mejor Cromosoma: {mejorCromosoma}")
    print(f"Mejor Aptitud: {mejorFuncObj:.6f}")
