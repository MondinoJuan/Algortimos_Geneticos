import random

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)] # Genera un cromosoma de ... genes aleatorios
    return cromosoma

def fitness(decimal, poblacion):
    total = 0
    for i in range(len(poblacion)):
        total += funcionObjetivo(binario_A_Decimal(poblacion[i]))

    return decimal / total

def funcionObjetivo(x):
    return (x * x) - 1   

def seleccionRandom(poblacion):
    padre1 = random.choice(poblacion)

    posible_padre2 = random.choice(poblacion)
    while padre1 == posible_padre2:
        posible_padre2 = random.choice(poblacion)

    padre2 = posible_padre2

    return padre1, padre2

def binario_A_Decimal(cromosoma):
    decimal = 0
    for i in range(len(cromosoma)):
        decimal += cromosoma[i] * (2 ** (len(cromosoma) - 1 - i))
    return decimal 

def mutar(individuo, prob_mutacion):            # Mutacion para probabilidades de hasta 0,0001
    for i in range(len(individuo)):
        
        if (prob_mutacion*10000 >= random.randint(1, 10000)):
            individuo[i] = cambiar_cromosoma(individuo[i])
    return individuo

def cambiar_cromosoma(cromosoma):
    if cromosoma == 0:
        cromosoma = 1
    else:
        cromosoma = 0
    return cromosoma



# Programa
# No usar macros ni librerias ni nada para el metodo de selección de padres.
corrida_random = []
corrida_elitismo = []
poblacion = []
cantidad = 10
cantGenes = 30
arregloFitness = []
cromosoma = []

# Generar la poblacion inicial
for i in range(cantidad):
    cromosoma = []
    cromosoma.append(completoCromosoma(cantGenes))
    poblacion.append(cromosoma)

print(poblacion[0])
print(poblacion[0][0])

#lista = [int, List[int], int, int, float]

# Calculo el fitness de un cromosoma
for individuo in poblacion:
    print(f"Individuo {i}: {individuo} -> Fitness: {fitness(funcionObjetivo(binario_A_Decimal(individuo[0])), poblacion)}")
    #arregloFitness.append(fitness(funcionObjetivo(cromosoma), poblacion))
    individuo.append(fitness(funcionObjetivo(individuo[0]), poblacion))

print(poblacion)



