##Tener descargado y en la misma carpeta el: Poligono Departamento Shapefile de la pagina: https://www.ign.gob.ar/NuestrasActividades/InformacionGeoespacial/CapasSIG
##Las coordenadas que da el mapa interactivo copiarlas en la variable geojson
import json
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon, LineString
from pathlib import Path 

def encontrar_departamento(*coords):

    # Carpeta donde está este script
    BASE_DIR = Path(__file__).resolve().parent  

    # Ruta al shapefile en la misma carpeta
    shapefile_path = BASE_DIR / "departamentoPolygon.shp"

    coordenadas = []

    for x in coords[0]:
        coordenadas.append(x)

    if len(coordenadas) == 1:
        geom = Point(coordenadas[0])
    elif len(coordenadas) == 2:
        geom = LineString(coordenadas)
    else:
        # Convertir a lista de listas
        lista_coords = [list(c) for c in coords[0]]
        # Asegurarse de cerrar el polígono (el primer punto = último punto)
        if lista_coords[0] != lista_coords[-1]:
            lista_coords.append(lista_coords[0])
        geom = Polygon(lista_coords)

    # Crear un GeoDataFrame con esa geometría
    poligono_gdf = gpd.GeoDataFrame(index=[0], geometry=[geom], crs="EPSG:4326")

    # Leer los departamentos desde el shapefile del IGN
    departamentos = gpd.read_file(shapefile_path)

    # Asegurarse que ambos estén en el mismo sistema de coordenadas (WGS84 - EPSG:4326)
    departamentos = departamentos.to_crs(epsg=4326)
    if geom.geom_type == "Point":
        interseccion = departamentos[departamentos.contains(geom)]
    elif geom.geom_type == "LineString":
        interseccion = departamentos[departamentos.intersects(geom)]
    else:
        # Hacer la intersección: encontrar qué departamentos intersectan con tu polígono
        interseccion = gpd.overlay(departamentos, poligono_gdf, how="intersection")
        # Evitar duplicados por fragmentos
        interseccion = interseccion.drop_duplicates(subset=["fna", "nam"])

    # Mostrar resultados
    if not interseccion.empty:
        deptos = []
        provs = []
        for _, row in interseccion.iterrows():
            provs.append(row['fna'])
            deptos.append(row['nam'])
            print(
                f"Provincia: {provs[-1]}, Departamento: {deptos[-1]}"
            )  ##ver porque a veces no sale la provincia
        return deptos[0], provs[0]
    else:
        print("El polígono no intersecta con ningún departamento.")
        return "No determinado", "No determinada"
                
#encontrar_departamento([(-60.274, -33.33763)])

#encontrar_departamento([[-60.275459,-33.287424],[-60.272498,-33.292156],[-60.266147,-33.287854],[-60.271597,-33.284914]])

#encontrar_departamento([(-60.275459,-33.287424),(-60.272498,-33.292156)])