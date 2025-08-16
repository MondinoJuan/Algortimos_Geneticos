import random
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
# Utiliza openpyxl tambien

# GENERAL
def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(cantidad):
    cromosoma = [aleatorio() for _ in range(cantidad)]
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes):
    poblacion = []
    for _ in range(cantidadCromosomas):
        cromosoma = completoCromosoma(cantidadGenes)
        poblacion.append(cromosoma)
    return poblacion

def binarioADecimal(cromosoma):
    decimal = 0
    exponente=0
    for i in range(len(cromosoma)-1,-1,-1):
        if cromosoma[i] == 1:
            decimal = decimal + pow(2,exponente)
        exponente += 1 
    return decimal

def funcionObjetivo(x):
    coef = (2 ** 30) - 1
    return (x / coef) ** 2  

def calculadorFuncionObjetivo(poblacion):
    objetivos = []
    for individuo in poblacion:
        decimal = binarioADecimal(individuo)
        obj = funcionObjetivo(decimal)
        objetivos.append(obj)
    return objetivos

def calculadorFitness(objetivos):
    fitness = []
    suma = sum(objetivos)
    for fo in objetivos:
        fit = fo / suma
        fitness.append(fit)
    return fitness

def calculadorEstadisticos(poblacion, objetivos):
    max_objetivos = max(objetivos)
    min_objetivos = min(objetivos)
    mejor_cromosoma = poblacion[objetivos.index(max_objetivos)]
    avg_objetivos = round((sum(objetivos)/len(objetivos)),4)
    return [max_objetivos,min_objetivos, avg_objetivos, mejor_cromosoma] 

# Crossover
def crossover1Punto(padre, madre):
    puntoCorte = random.randint(1, len(padre)-1)
    h1 = padre[:puntoCorte] + madre[puntoCorte:]
    h2 = madre[:puntoCorte] + padre[puntoCorte:]
    return h1, h2

# Mutacion
def mutacionInvertida(poblacion, probMutacion):
    for i in range(len(poblacion)):
        if random.random() < probMutacion:
            individuo = poblacion[i]
            pos1 = random.randint(0, len(individuo) - 1)
            pos2 = random.randint(0, len(individuo) - 1)
            # Ordenar para que pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            segmento_invertido = individuo[pos1:pos2+1][::-1]
            poblacion[i] = individuo[:pos1] + segmento_invertido + individuo[pos2+1:]
    return poblacion

# SELECCION
# Ruleta
def seleccionRuleta(poblacion, fitnessValores, cantidad):
    # Generar acumuladas
    acumuladas = []
    acum = 0
    for fit in fitnessValores:
        acum += fit
        acumuladas.append(acum)
    seleccionados = []
    for _ in range(cantidad):
        probAleatoria = random.random()
        for i, acum in enumerate(acumuladas):
            if probAleatoria <= acum:
                seleccionados.append(poblacion[i])
                break
    return seleccionados

# Torneo
def seleccionTorneo(poblacion, fitnessValores, cantidadIndividuos, cantidadCompetidores):
    ganadores = []
    for j in range(cantidadIndividuos):
        competidores = []
        fitness_competidores = []
        for i in range(cantidadCompetidores):
            indice = random.randint(0, len(poblacion)-1)
            competidores.append(poblacion[indice])
            fitness_competidores.append(fitnessValores[indice])
        ganador = competidores[fitness_competidores.index(max(fitness_competidores))]
        ganadores.append(ganador)
    return ganadores

# CICLOS
# Elitismo
def ciclos_con_elitismo(ciclos, prob_crossover, prob_mutacion, cant_individuos, cant_genes, metodo_seleccion, cantidadElitismo, cantidadCompetidores=None):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    
    pob = generarPoblacion(cantidadIndividuos,cant_genes) #Poblacion inicial random
    fo = calculadorFuncionObjetivo(pob)
    fit = calculadorFitness(fo)
    rta = calculadorEstadisticos(pob, fo)

    maximos.append(rta[0])
    minimos.append(rta[1])
    promedios.append(rta[2])
    mejores.append(rta[3])

    for j in range (ciclos):
        elitistas = [] 
        fit_ordenados = sorted(fit, reverse = True)

        for i in range(cantidadElitismo):
            indice = fit.index(fit_ordenados[i])
            elitistas.append(pob[indice])

        if metodo_seleccion == 'r':
            pob_intermedia = seleccionRuleta(pob, fit, cant_individuos - cantidadElitismo)
        else:
            pob_intermedia = seleccionTorneo(pob, fit, cant_individuos - cantidadElitismo, cantidadCompetidores) 
        #print(len(pob_intermedia))

        #REALIZO CROSSOVER Y MUTACION EN LA POBLACION
        for i in range (0,len(pob_intermedia),2):
            padre = pob_intermedia[i]
            madre = pob_intermedia[i+1]
            if random.random() < prob_crossover :
                hijo1, hijo2 = crossover1Punto(padre,madre)
                pob_intermedia[i], pob_intermedia[i+1] = hijo1, hijo2
            
        pob_intermedia = mutacionInvertida(pob_intermedia, prob_mutacion)
        
        pob = pob_intermedia + elitistas
        
        fo = calculadorFuncionObjetivo(pob)
        fit = calculadorFitness(fo)
        rta = calculadorEstadisticos(pob, fo)
        #GUARDAR VALORES NECESARIOS PARA LA GRAFICA
        maximos.append(rta[0])
        minimos.append(rta[1])
        promedios.append(rta[2])
        mejores.append(rta[3])
        
    return maximos, minimos, promedios, mejores

# Sin elitismo
def ciclos_sin_elitismo(ciclos, prob_crossover, prob_mutacion, cantidadIndividuos, cant_genes, metodo_seleccion, cantidadCompetidores=None):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    
    pob = generarPoblacion(cantidadIndividuos,cant_genes)
    fo = calculadorFuncionObjetivo(pob)
    fit = calculadorFitness(fo)
    rta = calculadorEstadisticos(pob, fo)
    
    maximos.append(rta[0])
    minimos.append(rta[1])
    promedios.append(rta[2])
    mejores.append(rta[3])

    for j in range (ciclos):
        if metodo_seleccion == 'r':
            pob = seleccionRuleta(pob, fit, cantidadIndividuos)
        else:
            pob = seleccionTorneo(pob, fit, cantidadIndividuos, cantidadCompetidores) 
        for i in range (0,len(pob),2):
            if i + 1 < len(pob):
                padre = pob[i]
                madre = pob[i+1]
            if random.random() < prob_crossover :
                hijo1, hijo2 = crossover1Punto(padre,madre)
                pob[i], pob[i+1] = hijo1, hijo2
        pob = mutacionInvertida(pob, prob_mutacion)
        fo = calculadorFuncionObjetivo(pob)
        fit = calculadorFitness(fo)
        rta = calculadorEstadisticos(pob, fo)

        #GUARDAR VALORES NECESARIOS PARA LA GRAFICA
        maximos.append(rta[0])
        minimos.append(rta[1])
        promedios.append(rta[2])
        mejores.append(rta[3])
    return maximos, minimos, promedios, mejores

# TABLAS EXCEL
def crear_tabla(maximos, minimos, promedios, mejores, metodo_seleccion, elitismo_Bool):
    cadenas = [''.join(str(num) for num in cromosoma) for cromosoma in mejores]
    decimales = [str(binarioADecimal(cromosoma)) for cromosoma in mejores]

    nombreMetodo = ''
    nombreElitismo = ''
    nombreCantidadCiclos = str(len(maximos)-1)

    if metodo_seleccion == 'r':
        nombreMetodo = '_Ruleta'
    else:
        nombreMetodo = '_Torneo'

    if elitismo_Bool == 1:
        nombreElitismo = '_Elitismo'

    df_nuevo = pd.DataFrame({
        'Corrida': range(len(maximos)),
        'Max': maximos,
        'Min': minimos,
        'AVG': promedios,
        'Decimal': decimales,
        'Mejor Cromosoma': cadenas,
    })

    archivo_excel = 'VALORES_' + nombreCantidadCiclos + 'Ciclos' + nombreMetodo + nombreElitismo + '.xlsx'

    if os.path.exists(archivo_excel):
        os.remove(archivo_excel)
        df_nuevo.to_excel(archivo_excel, index=False)
    else:
        df_nuevo.to_excel(archivo_excel, index=False)

# GRAFICOS
def generar_grafico(maximos, minimos, promedios, mejores, titulo, ciclo):
    x = list(range(len(maximos)))

    fig, ax = plt.subplots(figsize=(10, 6)) 

    ax.plot(x, maximos, label = 'Máximos',  marker='o', linestyle='-', color='b', linewidth=1, markersize=2)
    ax.plot(x, minimos, label = 'Mínimos', marker='o', linestyle='-', color='g', linewidth=1, markersize=2)
    ax.plot(x, promedios, label = 'Promedios', marker='o', linestyle='-', color='r', linewidth=1, markersize=2)

    ax.set_title('Máximos, Mínimos y Promedios')
    ax.set_xlabel('CORRIDA',fontsize=12)
    ax.set_ylabel('APTITUD',fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, ciclo + 2)
    ax.grid(True)
    ax.legend(fontsize = 10)

    fig.suptitle(titulo, fontsize=15)
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig(titulo.replace(" ", "_") + '.png')
    plt.show()

def verificar_maximo(datos):
    for i in range(1, len(datos)):
        if datos[i] < datos[i - 1]:
            print(f"Dato menor encontrado en índice {i}: {datos[i]} < {datos[i - 1]}")
            break
    else:
        print("Todos los datos son mayores o iguales a sus antecesores.")


# PROGRAMA PRINCIPAL
probCrossover = 0.75
probMutacion = 0.05
cantidadIndividuos = 10
cantidadElitismo = 2
cantidadCompetidores = int(cantidadIndividuos * 0.4)

cantidadGenes = 30
coef = (2 ** cantidadGenes) - 1
maximosPorCiclo = []
minimosPorCiclo = []
promediosPorCiclo = []

if len(sys.argv) != 7 or sys.argv[1] != "-c" or sys.argv[3] != "-s" or sys.argv[5] != "-e":
    print("Uso: python TP1_AG_G2.py -c <ciclos> -s <seleccion: r-ruleta t-torneo> -e <elitismo: 1-si 0-no>")
    sys.exit(1)
if int(sys.argv[2]) < 0 or (int(sys.argv[6]) != 0 and int(sys.argv[6]) != 1) or (sys.argv[4] != "r" and sys.argv[4] != "t"):
    print("Error: python TP1_AG_G2.py -c <ciclos> -s <seleccion: r-ruleta t-torneo> -e <elitismo: 1-si 0-no>")
    sys.exit(1)

ciclosPrograma = int(sys.argv[2])

if int(sys.argv[6]) == 1:
    maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_con_elitismo(ciclosPrograma,probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, metodo_seleccion=sys.argv[4], cantidadElitismo=cantidadElitismo, cantidadCompetidores=cantidadCompetidores)
    if sys.argv[4] == 'r':
        titulo = 'Seleccion RULETA ELITISTA - de '+ str(ciclosPrograma) + ' ciclos'
    else:
        titulo = 'Seleccion TORNEO ELITISTA - de '+ str(ciclosPrograma) + ' ciclos'
    generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclosPrograma)
    crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, sys.argv[4], int(sys.argv[6]))
else:
    maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_sin_elitismo(ciclosPrograma,probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, sys.argv[4], cantidadCompetidores=cantidadCompetidores)
    if sys.argv[4] == 'r':
        titulo = 'Seleccion RULETA - de '+ str(ciclosPrograma) + ' ciclos'
    else:
        titulo = 'Seleccion TORNEO - de '+ str(ciclosPrograma) + ' ciclos'
    generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclosPrograma)
    crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, sys.argv[4], int(sys.argv[6]))

verificar_maximo(maximosPorCiclo)