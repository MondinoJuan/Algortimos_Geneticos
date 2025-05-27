import random

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)] # Genera un cromosoma de ... genes aleatorios
    return cromosoma

def fitness(decimal, poblacion):
    total = 0
    for i in range(len(poblacion)):
        total += funcionObjetivo(binario_A_Decimal(poblacion[i]), 2 ** len(poblacion[i]) - 1)

    return decimal / total

def funcionObjetivo(x, coeficiente):
    return (x / coeficiente) ** 2  

'''
def seleccionRandom(poblacion):
    padre1 = random.choice(poblacion)

    posible_padre2 = random.choice(poblacion)
    while padre1 == posible_padre2:
        posible_padre2 = random.choice(poblacion)

    padre2 = posible_padre2

    return padre1, padre2
'''

def binario_A_Decimal(cromosoma):
    decimal = 0
    for i in range(len(cromosoma)):
        decimal += cromosoma[i] * (2 ** (len(cromosoma) - 1 - i))
    return decimal 

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

# Aleatoria
def seleccion_aleatoria(poblacion):
    padre1, padre2 = random.sample(poblacion, k=2)   # Elijo sin reemplazo, evito que se repita el mismo padre
    return padre1, padre2

# Por elitismo / torneo



# PROGRAMA
# No usar macros ni librerias ni nada para el metodo de selección de padres.
corrida_random = []
corrida_elitismo = []
poblacion = []
cantidadCromosomas = 10
cantGenes = 30
arregloFitness = []
#gen = []
probCrossOver = 0.75
probMutacion = 0.05
coeficiente = (2 ** cantGenes) - 1

# Generar la poblacion inicial
for i in range(cantidadCromosomas):
    cromosoma = []
    cromosoma = completoCromosoma(cantGenes)
    poblacion.append(cromosoma)

print(poblacion[0])
print(poblacion[0][0])

#lista = [int, List[int], int, int, float]

# Calculo el fitness de un cromosoma
for crom in poblacion:
    decimal = binario_A_Decimal(crom)
    fitness_value = fitness(decimal, poblacion)
    print(f"Cromosoma: {crom} -> Fitness: {fitness_value}")
    #arregloFitness.append(fitness(funcionObjetivo(gen), poblacion))                # No iría gen
    #cromosoma.append(fitness(funcionObjetivo(binario_A_Decimal(crom)), poblacion))         # A cromosoma le hago el append?

print(poblacion)



