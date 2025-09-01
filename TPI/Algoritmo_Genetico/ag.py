import random
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
# Utiliza openpyxl tambien

from Red_neuronal.My_GBM import main as utilizar_GBM

# GENERAL
'''
La idea es crear poblaciones con los mismos datos de suelo dependiendo del departamento y un clima predecido de aca a 14 meses,
diferenciandose en la semilla utilizada y el área a cultivar por semilla.
Deberia usarse el problema de la mochila planteado en el TP2?
'''

SEMILLAS = ['girasol', 'soja', 'maiz', 'trigo', 'sorgo', 'cebada', 'maní']

def aleatorio():
    return random.randint(0, 1)

def completoCromosoma(maximo, cantidad_genes=7):
    cromosoma = [random.random() for _ in range(cantidad_genes)]
    suma = sum(cromosoma)
    cromosoma = [(gen / suma) * maximo for gen in cromosoma]
    return cromosoma

def generarPoblacion(cantidadCromosomas, cantidadGenes, maximo):          
    poblacion = [] * cantidadCromosomas
    for _ in range(cantidadCromosomas):
        cromosoma = completoCromosoma(maximo, cantidadGenes)
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

def funcionObjetivo(x, precio):
    # Pasa por la red neuronal
    obj = x * precio  
    return obj

def calculadorFuncionObjetivo(poblacion): 
    objetivos = []

    # Recupero precios de la tonelada de semilla
    df_precios = pd.read_csv("Recuperacion_de_datos/Semillas/Archivos generados/precios_por_tonelada.csv")
    df_precios = df_precios.tail(7)
    df_precios = df_precios.iloc[:, [1, 2]]

    for individuo in poblacion:
        toneladas = red_neuronal(individuo)
        for idx, semilla in enumerate(SEMILLAS):
            precio = df_precios[df_precios.iloc[:, 0] == semilla].iloc[0, 1]
            obj = funcionObjetivo(toneladas[idx], precio)
        objetivos.append(obj)
    return objetivos

def red_neuronal(individuo, depto, lon, lat):
    # Uno con datos del suelo
    df_suelo = pd.read_csv("Recuperacion_de_datos/Suelos/suelo_promedio.csv")
    df_suelo = df_suelo[df_suelo['departamento_nombre'] == depto]

    # Uno con datos predecidos del clima
    from keras.models import load_model
    model = load_model("model/model.keras")
    predicciones_clima = model.predict(lon, lat)

    # Creo el dataframe final para pasar a la red neuronal
    df_final = pd.DataFrame()
    for area in individuo:
        df_final = df_final.append({
            'superficie_sembrada_ha': area,
            **df_suelo.iloc[0].to_dict(),
            **predicciones_clima.to_dict(), 
            'cultivo_nombre': SEMILLAS[individuo.index(area)],
            'anio': 2025
        }, ignore_index=True)

    cols = [['cultivo_nombre', 'anio', 'organic_carbon', 'ph', 'clay', 'silt', 'sand', 
                        'temperatura_media_C_1', 'temperatura_media_C_2', 'temperatura_media_C_3', 'temperatura_media_C_4', 
                        'temperatura_media_C_5', 'temperatura_media_C_6', 'temperatura_media_C_7', 'temperatura_media_C_8', 
                        'temperatura_media_C_9', 'temperatura_media_C_10', 'temperatura_media_C_11', 'temperatura_media_C_12', 
                        'temperatura_media_C_13', 'temperatura_media_C_14', 'humedad_relativa_%_1', 'humedad_relativa_%_2', 
                        'humedad_relativa_%_3', 'humedad_relativa_%_4', 'humedad_relativa_%_5', 'humedad_relativa_%_6', 
                        'humedad_relativa_%_7', 'humedad_relativa_%_8', 'humedad_relativa_%_9', 'humedad_relativa_%_10', 
                        'humedad_relativa_%_11', 'humedad_relativa_%_12', 'humedad_relativa_%_13', 'humedad_relativa_%_14', 
                        'velocidad_viento_m_s_1', 'velocidad_viento_m_s_2', 'velocidad_viento_m_s_3', 'velocidad_viento_m_s_4', 
                        'velocidad_viento_m_s_5', 'velocidad_viento_m_s_6', 'velocidad_viento_m_s_7', 'velocidad_viento_m_s_8', 
                        'velocidad_viento_m_s_9', 'velocidad_viento_m_s_10', 'velocidad_viento_m_s_11', 'velocidad_viento_m_s_12', 
                        'velocidad_viento_m_s_13', 'velocidad_viento_m_s_14', 'velocidad_viento_km_h_1', 'velocidad_viento_km_h_2', 
                        'velocidad_viento_km_h_3', 'velocidad_viento_km_h_4', 'velocidad_viento_km_h_5', 'velocidad_viento_km_h_6', 
                        'velocidad_viento_km_h_7', 'velocidad_viento_km_h_8', 'velocidad_viento_km_h_9', 'velocidad_viento_km_h_10', 
                        'velocidad_viento_km_h_11', 'velocidad_viento_km_h_12', 'velocidad_viento_km_h_13', 'velocidad_viento_km_h_14', 
                        'precipitacion_mm_mes_1', 'precipitacion_mm_mes_2', 'precipitacion_mm_mes_3', 'precipitacion_mm_mes_4', 
                        'precipitacion_mm_mes_5', 'precipitacion_mm_mes_6', 'precipitacion_mm_mes_7', 'precipitacion_mm_mes_8', 
                        'precipitacion_mm_mes_9', 'precipitacion_mm_mes_10', 'precipitacion_mm_mes_11', 'precipitacion_mm_mes_12', 
                        'precipitacion_mm_mes_13', 'precipitacion_mm_mes_14', 'superficie_sembrada_ha']]
    df_final = df_final[cols]
    
    predicciones_toneladas = utilizar_GBM(df_final)
    return predicciones_toneladas
    


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
def crossover1Punto(padre, madre, maximo):         
    puntoCorte = random.randint(1, len(padre)-1)
    h1 = padre[:puntoCorte] + madre[puntoCorte:]
    h2 = madre[:puntoCorte] + padre[puntoCorte:]
    h1 = normalizar(h1, maximo)
    h2 = normalizar(h2, maximo)
    return h1, h2

def normalizar(individuo, maximo):
    s = sum(individuo)
    return [(x/s) * maximo for x in individuo]

# Mutaciones
def mutacionInvertida(poblacion, probMutacion):         
    for i in range(len(poblacion)):
        if random.random() < probMutacion:
            individuo = poblacion[i]
            pos1 = random.randint(0, len(individuo) - 1)
            pos2 = random.randint(0, len(individuo) - 1)
            while pos1 == pos2:
                pos2 = random.randint(0, len(individuo) - 1)
            # Ordenar para que pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            segmento_invertido = individuo[pos1:pos2+1][::-1]
            poblacion[i] = individuo[:pos1] + segmento_invertido + individuo[pos2+1:]
    return poblacion

def mutacionSwap(poblacion, probMutacion):         
    # Se utilizará Swap mutation
    for i in range(len(poblacion)):
        if random.random() < probMutacion:
            individuo = poblacion[i]
            pos1 = random.randint(0, len(individuo) - 1)
            pos2 = random.randint(0, len(individuo) - 1)
            # Ordenar para que pos1 < pos2
            while pos1 == pos2:
                pos2 = random.randint(0, len(individuo) - 1)
            individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
            poblacion[i] = individuo
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
def ciclos_con_elitismo(ciclos, prob_crossover, prob_mutacion, cant_individuos, cant_genes, metodo_seleccion, cantidadElitismo, 
                        cantidadCompetidores=None):
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
def ciclos_sin_elitismo(ciclos, prob_crossover, prob_mutacion, cantidadIndividuos, cant_genes, metodo_seleccion, 
                        cantidadCompetidores=None):
    maximos=[]
    minimos=[]
    promedios=[]
    mejores=[]
    
    pob = generarPoblacion(cantidadIndividuos, cant_genes)
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



def main(depto, lat, lon, area_m2):
    probCrossover = 0.75
    probMutacion = 0.001
    cantidadIndividuos = 10
    cantidadElitismo = 2
    cantidadCompetidores = int(cantidadIndividuos * 0.4)

    cantidadGenes = 7
    maximosPorCiclo = []
    minimosPorCiclo = []
    promediosPorCiclo = []

    while True:
        ciclos = input("\nIngrese la cantidad de ciclos (debe ser un entero): ")
        try:
            ciclos = int(ciclos)
            break
        except ValueError:
            print("Por favor, ingrese un número entero válido.")
    while True:
        seleccion = input("\nIngrese el método de seleccion <r-ruleta t-torneo>: ")
        if seleccion.lower() in ['r', 't']:
            break
        else:
            print("Por favor, ingrese 'r' para ruleta o 't' para torneo.")
    while True:
        elitismo = input("\n¿Quiere usar elitismo? <elitismo: 1-si 0-no> ")
        if elitismo in ['0', '1']:
            elitismo = int(elitismo)
            break
        else:
            print("Por favor, ingrese '1' para sí o '0' para no.")
    
    if elitismo == 1:
        maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_con_elitismo(ciclos, probCrossover, probMutacion, 
                                                                                           cantidadIndividuos, cantidadGenes, 
                                                                                           seleccion, 
                                                                                           cantidadElitismo, cantidadCompetidores)
        if seleccion == 'r':
            titulo = 'Seleccion RULETA ELITISTA - de '+ str(ciclos) + ' ciclos'
        else:
            titulo = 'Seleccion TORNEO ELITISTA - de '+ str(ciclos) + ' ciclos'
        generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclos)
        crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, seleccion, elitismo)
    else:
        maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores = ciclos_sin_elitismo(ciclos, probCrossover, probMutacion, 
                                                                                           cantidadIndividuos, cantidadGenes, 
                                                                                           seleccion, cantidadCompetidores)
        if seleccion == 'r':
            titulo = 'Seleccion RULETA - de '+ str(ciclos) + ' ciclos'
        else:
            titulo = 'Seleccion TORNEO - de '+ str(ciclos) + ' ciclos'
        generar_grafico(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, titulo, ciclos)
        crear_tabla(maximosPorCiclo, minimosPorCiclo, promediosPorCiclo, mejores, seleccion, elitismo)



# PROGRAMA PRINCIPAL
probCrossover = 0.75
probMutacion = 0.05
cantidadIndividuos = 10
cantidadElitismo = 2
cantidadCompetidores = int(cantidadIndividuos * 0.4)

cantidadGenes = 7
maximosPorCiclo = []
minimosPorCiclo = []
promediosPorCiclo = []

verificar_maximo(maximosPorCiclo)