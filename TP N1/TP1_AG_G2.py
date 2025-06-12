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
    for i in range(cantidadCromosomas):
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
    return (x / coef) ** 2  

def calculadorEstadisticos(poblacion):
    objetivos = [funcionObjetivo(binarioADecimal(ind)) for ind in poblacion]
    max_objetivos = max(objetivos)
    min_objetivos = min(objetivos)
    mejor_cromosoma = poblacion[objetivos.index(max_objetivos)]
    avg_objetivos = round((sum(objetivos)/len(objetivos)),4)
    return [max_objetivos,min_objetivos, avg_objetivos, mejor_cromosoma]  

def calculadorFitness(poblacion):
    fitness = []
    objetivos = []
    sumatoria = 0

    #Calculamos los valores objetivos y los sumamos
    for individuo in poblacion: 
        decimal = binarioADecimal(individuo)
        obj = funcionObjetivo(decimal)
        objetivos.append(obj)
        sumatoria += obj
 
    if sumatoria == 0:
        print ('La suma de los valores es igual a cero.', poblacion)
        exit()
    
    #Guardo el peso de cada elemento de la poblacion. 
    for i in range (len(poblacion)):
        peso = objetivos[i]/sumatoria
        fitness.append(round(peso,5))
    return fitness

# Crossover
def crossover1Punto(padre, madre):
    puntoCorte = random.randint(1,len(padre)-1)
    h1 = padre[:puntoCorte] + madre[puntoCorte:]
    h2 = madre[:puntoCorte] + padre[puntoCorte:]
    return h1, h2

# Mutacion
def operadorMutacion(individuo):
    gen_a_mutar = random.randint(0,len(individuo)-1)
    individuo[gen_a_mutar] = 1 - individuo[gen_a_mutar]
    return individuo

# SELECCION
# Ruleta
def seleccionRuleta(poblacion, fitnessValores, cantidad):
    arregloRuleta = []
    seleccionados = []

    for i in range(len(fitnessValores)):
        fitnessValores[i] = fitnessValores[i]*100000
        fitnessValores[i] = int(fitnessValores[i])
        
        for j in range(fitnessValores[i]):
            arregloRuleta.append(poblacion[i]) 
    for i in range (cantidad):
            nro = random.randint(0,99999)
            seleccionados.append(arregloRuleta[nro])
    return seleccionados
        
# Torneo
def seleccionTorneo(poblacion, fitnessValores, cantidadIndividuos, cantidadCompetidores):
    ganadores = []

    for j in range(cantidadIndividuos):
        competidores = []
        fitness_competidores = []

        for i in range(cantidadCompetidores):
            c=random.randint(0,len(poblacion)-1)
            competidores.append(poblacion[c])
            fitness_competidores.append(fitnessValores[c])

        ganador = competidores[fitness_competidores.index(max(fitness_competidores))]
        ganadores.append(ganador)    
    
    return ganadores

# CICLOS
# Elitismo
def ciclos_con_elitismo(ciclos, prob_crossover, prob_mutacion,cantidadIndividuos, cantidadGenes, metodo_seleccion, cantidadElitismo, cantidadCompetidores=None):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    
    pob = generarPoblacion(cantidadIndividuos,cantidadGenes)
    fit = calculadorFitness(pob)
    rta = calculadorEstadisticos(pob)

    for j in range (ciclos):
        #De la poblacion me quedo con los de elite.
        elitistas = [] 
        fit_ordenados = sorted(fit, reverse=True)
        for i in range(cantidadElitismo):
            indice = fit.index(fit_ordenados[i])
            elitistas.append(pob[indice])

        if metodo_seleccion == 'r':
            pob = seleccionRuleta(pob,fit, cantidadIndividuos-cantidadElitismo)
        else:
            pob = seleccionTorneo(pob, fit, cantidadIndividuos-cantidadElitismo, cantidadCompetidores) 
        
        for i in range (0,len(pob),2):
            padre = pob[i]
            madre = pob[i+1]
            if random.random() < prob_crossover :
                hijo1, hijo2 = crossover1Punto(padre,madre)
                pob[i], pob[i+1] = hijo1, hijo2
            
            if random.random() < prob_mutacion: 
                pob[i] = operadorMutacion(pob[i])
            if random.random() < prob_mutacion: 
                pob[i+1] = operadorMutacion(pob[i+1])   

        pob = pob + elitistas
        fit = calculadorFitness(pob)
        rta = calculadorEstadisticos(pob)
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
    fit = calculadorFitness(pob)
    rta = calculadorEstadisticos(pob)
    
    maximos.append(rta[0])
    minimos.append(rta[1])
    promedios.append(rta[2])
    mejores.append(rta[3])

    for j in range (ciclos):
        if metodo_seleccion == 'r':
            pob = seleccionRuleta(pob,fit, cantidadIndividuos)
        else:
            pob = seleccionTorneo(pob, fit, cantidadIndividuos, cantidadCompetidores) 
        for i in range (0,len(pob),2):
            padre = pob[i]
            madre = pob[i+1]
            if random.random() < prob_crossover :
                hijo1, hijo2 = crossover1Punto(padre,madre)
                pob[i], pob[i+1] = hijo1, hijo2
            if random.random() < prob_mutacion: 
                pob[i] = operadorMutacion(pob[i])
            if random.random() < prob_mutacion: 
                pob[i+1] = operadorMutacion(pob[i+1])
        fit = calculadorFitness(pob)
        rta = calculadorEstadisticos(pob)
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

    fig, axs = plt.subplots(1, 3, figsize=(15, 4)) 

    axs[0].plot(x, maximos, marker='o', linestyle='-', color='b', linewidth=0.7, markersize=2)
    axs[0].set_title('Máximos')
    axs[0].set_xlabel('CORRIDA')
    axs[0].set_ylabel('APTITUD')
    axs[0].set_ylim(0, 1.2)
    axs[0].set_xlim(0, ciclo + 2)
    axs[0].grid(True)

    axs[1].plot(x, minimos, marker='o', linestyle='-', color='g', linewidth=0.7, markersize=2)
    axs[1].set_title('Mínimos')
    axs[1].set_xlabel('CORRIDA')
    axs[1].set_ylim(0, 1.2)
    axs[1].set_xlim(0, ciclo + 2)
    axs[1].grid(True)

    axs[2].plot(x, promedios, marker='o', linestyle='-', color='r', linewidth=0.7, markersize=2)
    axs[2].set_title('Promedios')
    axs[2].set_xlabel('CORRIDA')
    axs[2].set_ylim(0, 1.2)
    axs[2].set_xlim(0, ciclo + 2)
    axs[2].grid(True)

    fig.suptitle(titulo, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

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
    maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_con_elitismo(ciclosPrograma,probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, sys.argv[4], cantidadElitismo=cantidadElitismo, cantidadCompetidores=cantidadCompetidores)
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
