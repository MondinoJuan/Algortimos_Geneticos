import random
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
# Utiliza openpyxl

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
    coef = (2 ** 30) - 1
    return (x / coef) ** 2  

def calculadorEstadisticos(poblacion):
    objetivos = [funcionObjetivo(binarioADecimal(ind)) for ind in poblacion]
    max_objetivos = max(objetivos)
    min_objetivos = min(objetivos)
    mejor_cromosoma = poblacion[objetivos.index(max_objetivos)]
    prom_objetivos = round((sum(objetivos)/len(objetivos)),4)
    return [max_objetivos,min_objetivos, prom_objetivos, mejor_cromosoma]  

def calculadorFitness(poblacion):
    fitness = []
    objetivos = []
    sumatoria = 0

    for individuo in poblacion: 
        decimal = binarioADecimal(individuo)
        obj = funcionObjetivo(decimal)
        objetivos.append(obj)
        sumatoria += obj

    if sumatoria == 0:
        print ('La suma de los valores es igual a cero.', poblacion)
        exit()
    
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
def mutacionInvertida(hijo):
    cantGenes = len(hijo)
    if cantGenes < 2:
        return hijo  # No se puede invertir si hay menos de 2 genes
    posInicial = random.randint(0, cantGenes - 2)
    posFinal = random.randint(posInicial + 1, cantGenes - 1)
    segmento = hijo[posInicial:posFinal + 1]
    hijo[posInicial:posFinal + 1] = segmento[::-1]
    return hijo

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
        
        elitistas = [] 
        fitness_individuos = list(zip(fit, pob))
        fitness_individuos.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(cantidadElitismo):
            elitistas.append(fitness_individuos[i][1])
        
        pob_sin_elitistas = [ind for ind in pob if ind not in elitistas]
        
        if len(pob_sin_elitistas) < cantidadIndividuos - cantidadElitismo:
            pob_sin_elitistas = pob
        
        fit_sin_elitistas = calculadorFitness(pob_sin_elitistas)

        if metodo_seleccion == 'r':
            pob = seleccionRuleta(pob_sin_elitistas, fit_sin_elitistas, cantidadIndividuos-cantidadElitismo)
        else:
            pob = seleccionTorneo(pob_sin_elitistas, fit_sin_elitistas, cantidadIndividuos-cantidadElitismo, cantidadCompetidores) 
        
        for i in range (0,len(pob),2):
            padre = pob[i]
            if i+1 < len(pob):  # Verificar que existe la madre
                madre = pob[i+1]
                if random.random() < prob_crossover :
                    hijo1, hijo2 = crossover1Punto(padre,madre)
                    pob[i], pob[i+1] = hijo1, hijo2
                
                if random.random() < prob_mutacion: 
                    pob[i] = mutacionInvertida(pob[i])
                if random.random() < prob_mutacion: 
                    pob[i+1] = mutacionInvertida(pob[i+1])
            else:
                if random.random() < prob_mutacion: 
                    pob[i] = mutacionInvertida(pob[i])

        pob = elitistas + pob
        fit = calculadorFitness(pob)
        rta = calculadorEstadisticos(pob)
        
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
    
    pob = generarPoblacion(cantidadIndividuos, cant_genes)

    for j in range(ciclos):
        fit = calculadorFitness(pob)
        rta = calculadorEstadisticos(pob)
        maximos.append(rta[0])
        minimos.append(rta[1])
        promedios.append(rta[2])
        mejores.append(rta[3])

        if metodo_seleccion == 'r':
            pob = seleccionRuleta(pob, fit, cantidadIndividuos)
        else:
            pob = seleccionTorneo(pob, fit, cantidadIndividuos, cantidadCompetidores)

        for i in range(0, len(pob), 2):
            padre = pob[i]
            madre = pob[i+1]
            if random.random() < prob_crossover:
                hijo1, hijo2 = crossover1Punto(padre, madre)
                pob[i], pob[i+1] = hijo1, hijo2
            if random.random() < prob_mutacion:
                pob[i] = mutacionInvertida(pob[i])
            if random.random() < prob_mutacion:
                pob[i+1] = mutacionInvertida(pob[i+1])
    
    return maximos, minimos, promedios, mejores

# TABLAS EXCEL
def formatear_hoja_excel(hoja_trabajo):
    from openpyxl.styles import Font
    
    for columna in hoja_trabajo.columns:
        longitud_maxima = 0
        letra_columna = columna[0].column_letter
        for celda in columna:
            try:
                if len(str(celda.value)) > longitud_maxima:
                    longitud_maxima = len(str(celda.value))
            except:
                pass
        ancho_ajustado = min(longitud_maxima + 2, 50)
        hoja_trabajo.column_dimensions[letra_columna].width = ancho_ajustado
    
    for celda in hoja_trabajo[1]:
        celda.font = Font(bold=True)

def crear_tabla(maximos, minimos, promedios, mejores, metodo_seleccion, elitismo_Bool):
    cadenas = [''.join(str(num) for num in cromosoma) for cromosoma in mejores]
    decimales = [str(binarioADecimal(cromosoma)) for cromosoma in mejores]

    nombreMetodo = ''
    nombreElitismo = ''
    nombreCantidadCiclos = str(len(maximos))

    if metodo_seleccion == 'r':
        nombreMetodo = '_Ruleta'
    else:
        nombreMetodo = '_Torneo'

    if elitismo_Bool == 1:
        nombreElitismo = '_Elitismo'

    df_nuevo = pd.DataFrame({
        'Corrida': range(1, len(maximos) + 1),
        'Max': maximos,
        'Min': minimos,
        'AVG': promedios,
        'Decimal': decimales,
        'Mejor Cromosoma': cadenas,
    })

    archivo_excel = f'VALORES_{nombreCantidadCiclos}Ciclos_{nombreMetodo}_{nombreElitismo}.xlsx'
    metodo_corto = 'R' if metodo_seleccion == 'r' else 'T'
    elitismo_corto = 'E' if elitismo_Bool == 1 else ''
    nombre_hoja = f'{nombreCantidadCiclos}C_{metodo_corto}_{elitismo_corto}'
    if len(nombre_hoja) > 31:
        nombre_hoja = nombre_hoja[:31]

    if os.path.exists(archivo_excel):
        os.remove(archivo_excel)
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_nuevo.to_excel(writer, sheet_name=nombre_hoja, index=False)
        worksheet = writer.sheets[nombre_hoja]
        formatear_hoja_excel(worksheet)

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

# PROGRAMA PRINCIPAL
probCrossover = 0.75
probMutacion = 0.05
cantidadIndividuos = 10
cantidadElitismo = 2
cantidadCompetidores = int(cantidadIndividuos * 0.4)

cantidadGenes = 30
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
        titulo = 'Selección RULETA ELITISTA - '+ str(ciclosPrograma) + ' ciclos'
    else:
        titulo = 'Selección TORNEO ELITISTA - '+ str(ciclosPrograma) + ' ciclos'
    generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclosPrograma)
    crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, sys.argv[4], int(sys.argv[6]))
else:
    maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_sin_elitismo(ciclosPrograma,probCrossover, probMutacion, cantidadIndividuos, cantidadGenes, sys.argv[4], cantidadCompetidores=cantidadCompetidores)
    if sys.argv[4] == 'r':
        titulo = 'Selección RULETA - '+ str(ciclosPrograma) + ' ciclos'
    else:
        titulo = 'Selección TORNEO - '+ str(ciclosPrograma) + ' ciclos'
    generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclosPrograma)
    crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, sys.argv[4], int(sys.argv[6]))
