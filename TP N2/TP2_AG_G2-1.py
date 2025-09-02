import os
import sys
import random
import matplotlib.pyplot as plt
from itertools import combinations
from time import perf_counter

def limpiar_pantalla():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def ingreso_argumentos():
    if len(sys.argv) != 5 or sys.argv[1] != "-e" or sys.argv[3] != "-f":
        print("Uso: python TP2_AG_G2.py -e <eleccion de elementos: p-predefinido a-aleatorio, c-cargar por consola> -f <factor de decision: v-volumen p-peso>")
        sys.exit(1)
    if (sys.argv[2] != "p" and sys.argv[2] != "a" and sys.argv[2] != "c") or (sys.argv[4] != "v" and sys.argv[4] != "p"):
        print("Error: python TP2_AG_G2.py -e <eleccion de elementos: p-predefinido a-aleatorio, c-cargar por consola> -f <factor de decision: v-volumen p-peso>")
        sys.exit(1)
    return sys.argv[2], sys.argv[4]

def pedir_entero(mensaje, minimo, maximo):
    while True:
        try:
            valor = int(input(mensaje))
            if minimo <= valor <= maximo:
                return valor
            else:
                print(f"Error: Ingrese un valor entre {minimo} y {maximo}.")
                input("Presione enter para continuar...")
                print()
        except ValueError:
            print("Error: Ingrese un número entero.")
            input("Presione enter para continuar...")
            print()

def generar_elementos(cantidad_elementos, capacidad_maxima, precio_max_elem, es_aleatorio, factor_decision):
    elementos = [[] for _ in range(cantidad_elementos)]
    for i in range(cantidad_elementos):
        if es_aleatorio:
            medida = random.randint(1, capacidad_maxima)
            precio = random.randint(1, precio_max_elem)
        else:
            if factor_decision == "v":
                print(f"Ingrese el volumen del elemento {i + 1} (máximo {capacidad_maxima}): ", end="")
            else:
                print(f"Ingrese el peso del elemento {i + 1} (máximo {capacidad_maxima}): ", end="")
            medida = pedir_entero("", 1, capacidad_maxima)
            print(f"Ingrese el precio del elemento {i + 1} (máximo {precio_max_elem}): ", end="")
            precio = pedir_entero("", 1, precio_max_elem)
            print()
        elementos[i] = [i + 1, medida, precio]
    return elementos

def obtener_elementos_predefinidos(factor_decision):
    if factor_decision == "v":
        capacidad_maxima = 4200
        elementos = [
            #[numero, volumen, precio]
            [1, 150, 20],
            [2, 325, 40],
            [3, 600, 50],
            [4, 805, 36],
            [5, 430, 25],
            [6, 1200, 64],
            [7, 770, 54],
            [8, 60, 18],
            [9, 930, 46],
            [10, 353, 28]
        ]
    else:
        capacidad_maxima = 3000
        elementos = [
            #[numero, peso, precio]
            [1, 1800, 72],
            [2, 600, 36],
            [3, 1200, 60]
        ]
    return elementos, capacidad_maxima

def obtener_elementos_aleatorios(factor_decision):
    cant_elem_limite = 20
    cap_mochila_max = 10000
    precio_max_elem = 200
    cantidad_elementos = random.randint(1, cant_elem_limite)
    capacidad_maxima = random.randint(1, cap_mochila_max)
    elementos = generar_elementos(cantidad_elementos, capacidad_maxima, precio_max_elem, True, factor_decision)
    return elementos, capacidad_maxima

def obtener_elementos_consola(factor_decision):
    cant_elem_limite = 20
    cap_mochila_max = 10000
    precio_max_elem = 200
    cantidad_elementos = pedir_entero("Cantidad de objetos disponibles: ", 1, cant_elem_limite)
    if factor_decision == "v":
        capacidad_maxima = pedir_entero("Volumen máximo de la mochila: ", 1, cap_mochila_max)
    else:
        capacidad_maxima = pedir_entero("Peso máximo de la mochila: ", 1, cap_mochila_max)
    print()
    elementos = generar_elementos(cantidad_elementos, capacidad_maxima, precio_max_elem, False, factor_decision)
    input("Presione Enter para continuar...")
    limpiar_pantalla()
    
    return elementos, capacidad_maxima

def obtener_elementos_y_capacidad(eleccion_elementos, factor_decision):
    if eleccion_elementos == "p":
        return obtener_elementos_predefinidos(factor_decision)
    elif eleccion_elementos == "a":
        return obtener_elementos_aleatorios(factor_decision)
    else:
        return obtener_elementos_consola(factor_decision)

def mostrar_elementos(elementos, factor_decision):
    print()
    print("Elementos disponibles:")
    if factor_decision == "v":
        print(f"{'Número':<8} {'Volumen':<10} {'Precio':<8}")
    else:
        print(f"{'Número':<8} {'Peso':<10} {'Precio':<8}")
    print("-" * 28)
    for elem in elementos:
        print(f"{elem[0]:<8} {elem[1]:<10} {elem[2]:<8}")
    print()

#Búsqueda Exhaustiva
def generar_soluciones(elementos, capacidad):
    todas_soluciones = []
    numero_solucion = 0
    for r in range(len(elementos) + 1):
        for combinacion in combinations(elementos, r):
            numero_solucion += 1
            elementos_nums = [elem[0] for elem in combinacion]
            medida_total = sum(elem[1] for elem in combinacion)
            valor_total = sum(elem[2] for elem in combinacion)
            es_factible = medida_total <= capacidad
            
            todas_soluciones.append([
                numero_solucion,
                elementos_nums,
                medida_total,
                valor_total,
                es_factible
            ])
    return todas_soluciones

def graficar_distribucion_combinaciones(todas_soluciones, elementos, solucion_optima, titulo_archivo):
    max_elementos = len(elementos) + 1
    y_factibles = [0] * max_elementos
    y_no_factibles = [0] * max_elementos

    for solucion in todas_soluciones:
        num_elementos = len(solucion[1])
        es_factible = solucion[4]
        
        if es_factible:
            y_factibles[num_elementos] += 1
        else:
            y_no_factibles[num_elementos] += 1

    x = list(range(max_elementos))
    
    plt.figure(figsize=(12, 7))
    
    plt.bar(x, y_factibles, color='lightgreen', edgecolor='black', label='Soluciones Factibles')
    plt.bar(x, y_no_factibles, bottom=y_factibles, color='lightcoral', edgecolor='black', label='Soluciones No Factibles')
    
    num_elementos_optimo = len(solucion_optima[1])
    altura_marca = y_factibles[num_elementos_optimo] + y_no_factibles[num_elementos_optimo] + max(y_factibles + y_no_factibles) * 0.1
    
    plt.scatter(num_elementos_optimo, altura_marca, s=300, c='gold', marker='*', edgecolors='orange', linewidth=2, label=f'Solución Óptima (Valor: {solucion_optima[3]})', zorder=5)
    
    plt.annotate('', xy=(num_elementos_optimo, y_factibles[num_elementos_optimo] + y_no_factibles[num_elementos_optimo]), xytext=(num_elementos_optimo, altura_marca), arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))

    info_text = f"Solución Óptima:\n\nCantidad de elementos: {num_elementos_optimo}\nValor: {solucion_optima[3]}\nElementos: {solucion_optima[1]}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.5", edgecolor = 'orange', facecolor="gold", alpha=0.8), verticalalignment='top', fontsize=10, weight='bold')

    plt.title('Distribución de Combinaciones por Cantidad de Elementos (Búsqueda Exhaustiva)')
    plt.xlabel('Cantidad de Elementos por Combinación')
    plt.ylabel('Cantidad de Combinaciones')
    plt.legend()

    for i in range(len(x)):
        if y_factibles[i] > 0:
            plt.text(i, y_factibles[i]/2, str(y_factibles[i]), ha='center', va='center', color='darkgreen')
        if y_no_factibles[i] > 0:
            plt.text(i, y_factibles[i] + y_no_factibles[i]/2, str(y_no_factibles[i]), ha='center', va='center', color='darkred')

    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(titulo_archivo + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def busqueda_exhaustiva(elementos, capacidad):
    tiempo_inicio_exh = perf_counter()

    todas_soluciones = generar_soluciones(elementos, capacidad)

    soluciones_factibles = [sol for sol in todas_soluciones if sol[4] == True]

    solucion_optima = max(soluciones_factibles, key=lambda x: x[3]) if soluciones_factibles else None
    tiempo_fin_exh = perf_counter()
    demora_exh = tiempo_fin_exh - tiempo_inicio_exh
    
    return solucion_optima, demora_exh, todas_soluciones, soluciones_factibles

#Búsqueda Heurística
def algoritmo_greedy(elementos:list[list], capacidad_max):
    tiempo_inicio_greedy = perf_counter()

    elementos_greedy = []
    for elem in elementos:
        numero, medida, valor = elem
        relacion = valor/medida
        elementos_greedy.append([numero, medida, valor, relacion])
    elementos_greedy.sort(key=lambda x: x[3], reverse=True)
    
    mochila = []
    sum_capacidad_mochila = 0
    sum_precio_mochila = 0
    capacidad_restante = capacidad_max

    for elem in elementos_greedy:
        numero, medida, valor, relacion = elem
        if medida <= capacidad_restante:
            sum_capacidad_mochila += medida
            sum_precio_mochila += valor
            capacidad_restante -= medida
            mochila.append(elem)

    tiempo_fin_greedy = perf_counter()
    demora_greedy = tiempo_fin_greedy - tiempo_inicio_greedy

    return mochila, sum_precio_mochila, sum_capacidad_mochila, demora_greedy

def mostrar_resultados_exhaustiva(elementos, solucion_optima, demora, todas_soluciones, soluciones_factibles, factor_decision):
    tipo_busqueda = "Volumen" if factor_decision == "v" else "Peso"
    unidad = "cm^3" if factor_decision == "v" else "gramos"
    nombre_archivo = "busq_exh_vol_graf" if factor_decision == "v" else "busq_exh_peso_graf"
    print(f"Búsqueda Exhaustiva en Función del {tipo_busqueda}")
    graficar_distribucion_combinaciones(todas_soluciones, elementos, solucion_optima, nombre_archivo)
    print(f"Precio máximo (Exhaustiva): $ {solucion_optima[3]} ; {tipo_busqueda}: {solucion_optima[2]} {unidad}")
    print(f"Elementos en mochila (Exhaustiva): {solucion_optima[1]}")
    print(f"Tiempo de demora (Exhaustiva): {demora:.6f} segundos")
    print(f"Cantidad de soluciones: {len(todas_soluciones)}\nCantidad de soluciones factibles: {len(soluciones_factibles)}")

def mostrar_resultados_greedy(mochila_greedy, precio, medida, demora, factor_decision):
    tipo_busqueda = "Volumen" if factor_decision == "v" else "Peso"
    unidad = "cm^3" if factor_decision == "v" else "gramos"
    print(f"Búsqueda Heurística (Algoritmo Greedy) en Función del {tipo_busqueda}")
    print(f"Precio obtenido (Greedy): $ {precio} ; {tipo_busqueda}: {medida} {unidad}")
    print(f"Elementos en mochila (Greedy): {[elem[0] for elem in mochila_greedy]}")
    print(f"Tiempo de demora (Greedy): {demora:.6f} segundos")

#Programa principal
limpiar_pantalla()
eleccion_elementos, factor_decision = ingreso_argumentos()
elementos, capacidad_maxima = obtener_elementos_y_capacidad(eleccion_elementos, factor_decision)
mostrar_elementos(elementos, factor_decision)
mochila_exh, demora_exh, todas_soluciones, soluciones_factibles = busqueda_exhaustiva(elementos, capacidad_maxima)
mochila_greedy, precio, medida, demora_greedy = algoritmo_greedy(elementos, capacidad_maxima)
mostrar_resultados_exhaustiva(elementos, mochila_exh, demora_exh, todas_soluciones, soluciones_factibles, factor_decision)
print()
print("-" * 74)
print()
mostrar_resultados_greedy(mochila_greedy, precio, medida, demora_greedy, factor_decision)
print()
