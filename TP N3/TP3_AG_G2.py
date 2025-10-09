import os
import time
from capitales import obtenerCapitales, mostrarCapitales, mostrarCapital, obtenerDistancias, mostrarDistanciasParciales, visualizarRecorrido
from busquedaHeuristica import busquedaHeuristica, mejorRecorridoHeuristica
from algoritmoGenetico import algoritmoGenetico

def limpiarPantalla():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def pausa():
    input("\nPresione Enter para continuar...")

def mostrarTitulo():
    print("-"*100)
    print("\t"*5+"PROBLEMA DE LA MOCHILA")
    print("-"*100+"\n")

def submenuCapital(capitales):
    while True:
        limpiarPantalla()
        print("Capitales provinciales:")
        mostrarCapitales(capitales)
        try:
            opcion = int(input("\nSeleccione una capital de inicio (1-24): "))
            if 1 <= opcion <= 24:
                limpiarPantalla()
                indiceOrigen = opcion - 1
                return indiceOrigen
            else:
                print(f"\nIngrese un número entero entre 1 y {len(capitales)}")
                pausa()
        except ValueError:
            print(f"\nError: Ingrese un número entero válido")
            pausa()

def menuAlgGen(capitales, distancias):
    opcionElitismo = None
    opcionSeleccion = None
    while opcionElitismo is None:
        limpiarPantalla()
        print("1. Realizar Algoritmo Genético Sin Elitismo")
        print("2. Realizar Algoritmo Genético Con Elitismo")
        print("0. Volver al menú principal")
        try:
            opcion = int(input("\nSeleccione si el Algoritmo Genético se realiza con o sin Elitismo (0-2): "))
            if opcion == 1 or opcion == 2:
                opcionElitismo = opcion
            elif opcion == 0:
                limpiarPantalla()
                return
            else:
                print("\nIngrese un número entero entre 0 y 2")
                pausa()
        except ValueError:
            print("\nError: Ingrese un número entero válido")
            pausa()
    while opcionSeleccion is None:
        limpiarPantalla()
        print("1. Realizar Algoritmo Genético con selección mediante Ruleta")
        print("2. Realizar Algoritmo Genético con selección mediante Torneo")
        print("3. Volver al menú anterior")
        print("0. Volver al menú principal")
        try:
            opcion = int(input("\nSeleccione si la selección se realiza mediante Ruleta o Torneo (0-3): "))
            if opcion == 1 or opcion == 2:
                opcionSeleccion = opcion
            elif opcion == 3:
                return menuAlgGen(capitales, distancias)
            elif opcion == 0:
                limpiarPantalla()
                return
            else:
                print("\nIngrese un número entero entre 0 y 3")
                pausa()
        except ValueError:
            print("\nError: Ingrese un número entero válido")
            pausa()
    limpiarPantalla()
    recorrido, distanciasParciales, distanciaRecorrida, demora = algoritmoGenetico(opcionElitismo, opcionSeleccion, distancias)
    indiceOrigen = recorrido[0]
    mostrarResultado(capitales, indiceOrigen, distanciasParciales, recorrido, distanciaRecorrida, demora, 3)
    pausa()

def mostrarResultado(capitales, indiceOrigen, distanciasParciales, recorrido, distanciaRecorrida, demora, opcion):
    print(f"\nOrigen: {mostrarCapital(capitales, indiceOrigen)}")
    mostrarCapital(capitales, indiceOrigen)
    print(f"\nDistancias parciales:")
    mostrarDistanciasParciales(distanciasParciales, capitales, recorrido)
    print(f"Distancia total: {distanciaRecorrida} km")
    print(f"\nTiempo de demora: {demora:.6f} segundos")
    visualizarRecorrido(recorrido, capitales, opcion)

def menu():
    capitales = obtenerCapitales()
    distancias = obtenerDistancias()
    while True:
        limpiarPantalla()
        mostrarTitulo()
        print("1. Obtener ruta mínima desde un origen mediante una Búsqueda Heurística")
        print("2. Obtener ruta mínima mediante una Búsqueda Heurística")
        print("3. Obtener ruta mínima mediante un Algoritmo Genético")
        print("0. Salir")
        try:
            opcion = int(input("\nSeleccione una opción (0-3): "))
            if opcion == 1:
                indiceOrigen = submenuCapital(capitales)
                recorrido, distanciasParciales, distanciaRecorrida, demora = busquedaHeuristica(distancias, indiceOrigen, opcion)
                mostrarResultado(capitales, indiceOrigen, distanciasParciales, recorrido, distanciaRecorrida, demora, opcion)
                pausa()
            elif opcion == 2:
                limpiarPantalla()
                recorrido, distanciasParciales, distanciaRecorrida, indiceOrigen, demora = mejorRecorridoHeuristica(distancias)
                mostrarResultado(capitales, indiceOrigen, distanciasParciales, recorrido, distanciaRecorrida, demora, opcion)
                pausa()
            elif opcion == 3:
                menuAlgGen(capitales, distancias)
            elif opcion == 0:
                limpiarPantalla()
                print("¡Hasta luego!")
                time.sleep(5)
                limpiarPantalla()
                break
            else:
                print("\nIngrese un número entero entre 0 y 3")
                pausa()
        except ValueError:
            print("\nError: Ingrese un número entero válido")
            pausa()

#! Programa Principal
menu()