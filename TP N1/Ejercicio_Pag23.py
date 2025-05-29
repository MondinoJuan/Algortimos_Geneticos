import random

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)] # Genera un cromosoma de ... genes aleatorios
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes):
    for i in range(cantidadCromosomas):
        cromosoma = []
        cromosoma = completoCromosoma(cantidadGenes)
        poblacion.append(cromosoma)
    return poblacion

'''
def fitness(decimal, poblacion, cantGenes):
    total = 0
    for i in range(len(poblacion)):
        total += funcionObjetivo(binarioADecimal(poblacion[i], cantGenes), 2 ** len(poblacion[i]) - 1)

    return decimal / total
'''
    
def funcionObjetivo(x, coef):
    return (x / coef) ** 2  

def binarioADecimal(cromosoma, cantGenes):
    decimal = 0
    for i in range(cantGenes):
        decimal += cromosoma[i] * (2 ** (cantGenes - 1 - i))
    return decimal 

def crossoverUnPunto(cantGenes, padres):
    lugarCorte = random.randint(1, cantGenes - 1)
    hijoUno = []
    hijoDos = []
    for i in range(lugarCorte):
        hijoUno.append(padres[0][i])
        hijoDos.append(padres[1][i])
    for i in range(lugarCorte, cantGenes):
        hijoUno.append(padres[1][i])
        hijoDos.append(padres[0][i])
    hijos = [hijoUno, hijoDos]
    return hijos

def mutar(cromosoma, prob_mutacion):            # Mutacion para probabilidades de hasta 0,0001
    for i in range(len(cromosoma)):
        
        if (prob_mutacion*10000 >= random.randint(1, 10000)):
            cromosoma[i] = cambiar_gen(cromosoma[i])
    return cromosoma

def cambiar_gen(gen):
    if gen == 0:
        gen = 1
    else:
        gen = 0
    return gen

# SELECCION
# Por ruleta
def seleccion_ruleta(poblacion):
    pass

# Aleatoria
def seleccion_aleatoria(poblacion):
    padres = random.sample(poblacion, k=2)   # Elijo sin reemplazo, evito que se repita el mismo padre
    return padres
# Por elitismo
def seleccion_elitismo(poblacion):
    pass

# Por torneo
def seleccion_torneo(poblacion):
    pass

'''
def seleccionRandom(poblacion):
    padre1 = random.choice(poblacion)

    posible_padre2 = random.choice(poblacion)
    while padre1 == posible_padre2:
        posible_padre2 = random.choice(poblacion)

    padre2 = posible_padre2

    return padre1, padre2
'''

# PROGRAMA
# No usar macros ni librerias ni nada para el metodo de selección de padres.
probCrossover = 0.75
probMutacion = 0.05
cantidadCromosomas = 10
ciclosPrograma = 1 #Cambiar luego a 20
cantGenes = 30
coef = (2 ** cantGenes) - 1
poblacion = []
arregloFitness = []
corrida_random = []
corrida_elitismo = []

# Generar la poblacion inicial
poblacion = generarPoblacion(cantidadCromosomas, cantGenes)

for crom in poblacion:
    print(crom)
print()

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

print("----Padres----")
padres = seleccion_aleatoria(poblacion) #Selección de prueba
for padre in padres:
    print(f"Cromosoma: {padre}")
print()    

print("----Hijos----")
if random.uniform(0, 1) <= probCrossover:
    hijos = crossoverUnPunto(cantGenes, padres)
for hijo in hijos:
    print(f"Cromosoma: {hijo}")
print()