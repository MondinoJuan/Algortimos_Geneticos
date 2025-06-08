import random

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)] # Genera un cromosoma de ... genes aleatorios
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes):
    poblacion = []
    for i in range(cantidadCromosomas):
        cromosoma = []
        cromosoma = completoCromosoma(cantidadGenes)
        poblacion.append(cromosoma)
    return poblacion
    
def funcionObjetivo(x, coef):
    return (x / coef) ** 2  

def binarioADecimal(cromosoma, cantGenes):
    decimal = 0
    for i in range(cantGenes):
        decimal += cromosoma[i] * (2 ** (cantGenes - 1 - i))
    return decimal 

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

def mutacionInvertida(cantGenes, hijo):
    if cantGenes < 2:
        return hijo  # No se puede invertir si hay menos de 2 genes
    posInicial = random.randint(0, cantGenes - 2)
    posFinal = random.randint(posInicial + 1, cantGenes - 1)
    segmento = hijo[posInicial:posFinal + 1]
    hijo[posInicial:posFinal + 1] = segmento[::-1]  # Invertir el segmento
    return hijo

# SELECCION
# Aleatoria
def seleccionAleatoria(poblacion):
    padres = random.sample(poblacion, k=2)   # Elijo sin reemplazo, evito que se repita el mismo padre
    return padres

# Ruleta
def seleccionRuleta(poblacion, fitnessValores):
    probAcumulativas = []
    probAcum = 0
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

# PROGRAMA
# No usar macros ni librerias ni nada para el metodo de selección de padres.
probCrossover = 0.75
probMutacion = 0.99 #Cambiar luego a 0.05
cantidadCromosomas = 10
ciclosPrograma = 1 #Cambiar luego a 20
cantGenes = 10 #Cambiar luego a 30
coef = (2 ** cantGenes) - 1
corrida_random = []
corrida_elitismo = []

# Generar la poblacion inicial
poblacion = []
poblacion = generarPoblacion(cantidadCromosomas, cantGenes)
print("----Población Inicial----")

#Solo una corrida

# Calculo el fitness de cada cromosoma
funcObjValores = []
fitnessValores = []
suma = 0
for crom in poblacion:
    decimal = binarioADecimal(crom, cantGenes)
    funcObjValor = funcionObjetivo(decimal, coef)
    suma += funcObjValor
    funcObjValores.append(funcObjValor)

for crom, funcObjValor in zip(poblacion, funcObjValores):
    fitnessValor = funcObjValor / suma
    fitnessValores.append(fitnessValor)
    print(f"Cromosoma: {crom} -> Fitness: {fitnessValor}")
print()

print("----Selección----")
#seleccionRuleta(poblacion, fitnessValores)
padres = seleccionAleatoria(poblacion) #Selección de prueba
for padre in padres:
    print(f"Cromosoma: {padre}")
print()    

print("----Crossover----")
if random.uniform(0, 1) <= probCrossover:
    hijos = crossover1Punto(cantGenes, padres)
else:
    hijos = padres
for hijo in hijos:
    print(f"Cromosoma: {hijo}")
print()

print("----Mutaciones----")
for i in range(len(hijos)):
    if random.uniform(0, 1) <= probMutacion:
        hijos[i] = mutacionInvertida(cantGenes, hijos[i])
i = 0
for hijo in hijos:
    print(f"Mutación en el hijo {i + 1}: {hijo}")
    i += 1
print()