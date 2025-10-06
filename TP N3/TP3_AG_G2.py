import os
import time
from capitales import obtenerCapitales, mostrarCapitales, mostrarCapital, obtenerDistancias

def limpiarPantalla():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def pausa():
    input("\nPresione Enter para continuar...")

def submenuCapital(capitales):
    while True:
        limpiarPantalla()
        print("LISTA DE CAPITALES PROVINCIALES")
        mostrarCapitales(capitales)
        try:
            opcion = int(input("\nSeleccione una capital de inicio (1-24): "))
            if 1 <= opcion <= 24:
                limpiarPantalla()
                print("\nCAPITAL INICIAL SELECCIONADA")
                indexCapital = opcion - 1
                mostrarCapital(capitales, indexCapital)
                input("\nPresione Enter para continuar...")
                limpiarPantalla()
                break
            else:
                print(f"\nIngrese un número entero entre 1 y {len(capitales)}")
                pausa()
        except ValueError:
            print(f"\nError: Ingrese un número entero válido")
            pausa()


def submenuAlgGen():
    while True:
        limpiarPantalla()
        print("ALGORITMO GENÉTICO")
        print("1. Algoritmo Genético Sin Elitismo")
        print("2. Algoritmo Genético Con Elitismo")
        print("0. Volver al Menú Principal")
        try:
            opcion = int(input("\nSeleccione una opción (0-2): "))
            if opcion == 1:
                print("Funcionalidad en construcción")
                pausa()
            elif opcion == 2:
                print("Funcionalidad en construcción")
                pausa()
            elif opcion == 0:
                limpiarPantalla()
                break
            else:
                print("\nIngrese un número entero entre 0 y 2")
                pausa()
        except ValueError:
            print("\nError: Ingrese un número entero válido")
            pausa()

def menu():
    capitales = obtenerCapitales()
    distancias = obtenerDistancias()
    while True:
        limpiarPantalla() 
        print("-"*100)
        print("\t"*5+"PROBLEMA DE LA MOCHILA")
        print("-"*100+"\n")
        print("1. Obtener ruta mínima desde un origen mediante una Búsqueda Heurística")
        print("2. Obtener ruta mínima mediante una Búsqueda Heurística")
        print("3. Obtener ruta mínima mediante un Algoritmo Genético")
        print("0. Salir")
        try:
            opcion = int(input("\nSeleccione una opción (0-3): "))
            if opcion == 1:
                submenuCapital(capitales)
                print("Funcionalidad en construcción")
                pausa()
            elif opcion == 2:
                limpiarPantalla()
                print("Funcionalidad en construcción")
                pausa()
            elif opcion == 3:
                limpiarPantalla()
                print("Funcionalidad en construcción")
                pausa()
            elif opcion == 0:
                limpiarPantalla()
                print("\n¡Hasta luego!")
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