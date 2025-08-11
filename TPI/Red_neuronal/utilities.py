import pandas as pd
from math import radians, sin, cos, sqrt, asin
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import obtener_datos_nasa_power, procesar_datos_mensuales
from Recuperacion_de_datos.Suelos.Recupero_suelos_de_tablas import recupero_datos_suelo
from Recuperacion_de_datos.Semillas.Recupero_semillas import recupero_datos_avena, recupero_datos_cebada, recupero_datos_centeno, recupero_datos_girasol, recupero_datos_maiz, recupero_datos_mani, recupero_datos_mijo, recupero_datos_soja, recupero_datos_trigo

# Recupero los datos climáticos a partir de unas coordenadas.
# ---------------------------------------------------------------------------------------------------------------------
def climate_for_coordinates(latitud, longitud):
    años_atras = 44
    
    print("=== OBTENIENDO DATOS DIARIOS Y PROCESANDO MENSUALMENTE ===")
    # Obtener datos diarios
    df_clima_diario = obtener_datos_nasa_power(latitud, longitud, años_atras)
    
    if not df_clima_diario.empty:
        # Procesar a datos mensuales
        df_clima = procesar_datos_mensuales(df_clima_diario)
        print(f"✅ Datos diarios procesados a {len(df_clima)} registros mensuales")
    
    if not df_clima.empty:
        # Guardar en CSV
        output_path = f"Recuperacion_de_datos/Clima/clima_nasa_mensual_{años_atras}_anios.csv"
        df_clima.to_csv(output_path, index=False)    
    else:
        print("\n❌ No se pudieron obtener datos de NASA POWER")
        print("Verifica las coordenadas y la conexión a internet")


# Recupero los datos del suelo a partir de las coordenadas más cercanas a la ubicación del campo.
# ---------------------------------------------------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia haversine entre dos puntos en la Tierra
    Argumentos en grados decimales, retorna distancia en kilómetros

    Fórmula Haversine: calcula la distancia entre dos puntos en la superfice terrestre 
        considerando la curvatura de la tierra.
    """
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Fórmula haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radio de la Tierra en kilómetros
    r = 6371
    
    return c * r

def suelo_for_coordinates(latitud, longitud, max_distance=None):
    """
    Encuentra los datos de suelo más cercanos a las coordenadas dadas
    
    Args:
        latitud: Latitud objetivo
        longitud: Longitud objetivo  
        max_distance: Distancia máxima en km (opcional, para filtrar resultados muy lejanos)
    
    Returns:
        DataFrame con los datos del punto más cercano y la distancia
    """
    recupero_datos_suelo()

    suelo = pd.read_csv('Recuperacion de datos/Suelos/suelo_unido.csv', encoding='latin1')
    
    # Calcular la distancia de cada punto a las coordenadas objetivo
    suelo['distancia_km'] = suelo.apply(
        lambda row: haversine_distance(latitud, longitud, row['latitud'], row['longitud']), 
        axis=1
    )
    
    # Filtrar por distancia máxima si se especifica
    if max_distance is not None:
        suelo = suelo[suelo['distancia_km'] <= max_distance]
        
        if suelo.empty:
            print(f"No se encontraron datos dentro de {max_distance} km de las coordenadas: {latitud}, {longitud}")
            return pd.DataFrame()
    
    # Encontrar el punto más cercano
    punto_mas_cercano = suelo.loc[suelo['distancia_km'].idxmin()]
    
    print(f"Punto más cercano encontrado a {punto_mas_cercano['distancia_km']:.2f} km de distancia")
    print(f"Coordenadas del punto: {punto_mas_cercano['latitud']}, {punto_mas_cercano['longitud']}")
    
    # Crear DataFrame con todos los datos del punto más cercano
    resultado = pd.DataFrame([punto_mas_cercano])
    
    return resultado


# Recupero los datos de semillas y producción histórica a partir de un departamento.
# ---------------------------------------------------------------------------------------------------------------------
def seeds_for_department(department_name, seed_name):

    # Cargar datos de semillas me aseguro de tener el archivo correspondiente creado.
    match seed_name:
        case 'avena':
            recupero_datos_avena()
        case 'cebada':
            recupero_datos_cebada()
        case 'centeno':
            recupero_datos_centeno()
        case 'girasol':
            recupero_datos_girasol()
        case 'maiz':
            recupero_datos_maiz()
        case 'mani':
            recupero_datos_mani()
        case 'mijo':
            recupero_datos_mijo()
        case 'soja':
            recupero_datos_soja()
        case 'trigo':
            recupero_datos_trigo()
        

    seed = pd.read_csv(f'Recuperacion_de_datos/Semillas/Archivos generados/{seed_name}_recuperado.csv', encoding='latin1')

    # Filtro por nombre del departamento
    seed_filtered = seed[seed['departamento_nombre'] == department_name]

    if seed_filtered.empty:
        print(f"No se encontraron datos para el departamento: {department_name}")
        return pd.DataFrame()
    
    return seed_filtered