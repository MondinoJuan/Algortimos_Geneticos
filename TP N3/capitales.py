import pandas as pd

def obtenerCapitales():
    return [
        ["Ciudad Autónoma de Buenos Aires", "CABA", -34.6037, -58.3816],
        ["Córdoba", "Córdoba", -31.4201, -64.1888],
        ["Corrientes", "Corrientes", -27.4691, -58.8306], 
        ["Formosa", "Formosa", -26.1775, -58.1781],
        ["La Plata", "Buenos Aires", -34.9214, -57.9544],
        ["La Rioja", "La Rioja", -29.4144, -66.8558],
        ["Mendoza", "Mendoza", -32.8895, -68.8458],
        ["Neuquén", "Neuquén", -38.9516, -68.0591],
        ["Paraná", "Entre Ríos", -31.7277, -60.5290],
        ["Posadas", "Misiones", -27.3668, -55.8961],
        ["Rawson", "Chubut", -43.3000, -65.1000],
        ["Resistencia", "Chaco", -27.4517, -58.9867],
        ["Río Gallegos", "Santa Cruz", -51.6230, -69.2168],
        ["San Fernando del Valle de Catamarca", "Catamarca", -28.4696, -65.7852],
        ["San Miguel de Tucumán", "Tucumán", -26.8304, -65.2038],
        ["San Salvador de Jujuy", "Jujuy", -24.1858, -65.2995],
        ["Salta", "Salta", -24.7886, -65.4104],
        ["San Juan", "San Juan", -31.5373, -68.5257],
        ["San Luis", "San Luis", -33.302273, -66.336877],
        ["Santa Fe", "Santa Fe", -31.6466, -60.7099],
        ["Santa Rosa", "La Pampa", -36.6200, -64.2900],
        ["Santiago del Estero", "Santiago del Estero", -27.7876, -64.2596],
        ["Ushuaia", "Tierra del Fuego", -54.8073, -68.3037],
        ["Viedma", "Río Negro", -40.8135, -63.0000]
    ]

def mostrarCapitales(capitales):
    for i, [capital, provincia, _, _] in enumerate(capitales, 1):
        if (i == 0):
            print(f"{i:2d}. {capital:<25}")
        else:
            print(f"{i:2d}. {capital:<25} → {provincia}")

def mostrarCapital(capitales, indexCapital):
    capital, provincia, _, _ = capitales[indexCapital]
    if (indexCapital == 0):
        print(f"\tCapital: {capital}")
    else:
        print(f"\tCapital: {capital}")
        print(f"\tProvincia: {provincia}")

def obtenerDistancias():
    df = pd.read_excel('TablaCapitales.xlsx')
    return df.iloc[:24, :25]