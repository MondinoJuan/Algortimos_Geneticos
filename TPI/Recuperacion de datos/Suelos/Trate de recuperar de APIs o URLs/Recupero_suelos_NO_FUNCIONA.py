import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import json

def obtener_capacidades_wms_inta():
    """
    Obtiene las capacidades del servicio WMS del INTA para conocer las capas disponibles
    """
    # URLs conocidas de servicios WMS del INTA
    servicios_wms = [
        "https://geointa.inta.gob.ar/geoserver/wms",
        "https://geo.inta.gob.ar/geoserver/wms",
        "https://ideg.inta.gob.ar/geoserver/wms",
        "https://sig.inta.gob.ar/geoserver/wms",

        "https://geointa.inta.gob.ar/geoserver/wms",
        "https://visor.geointa.inta.gob.ar/geoserver/wms",
        "http://www.geointa.inta.gob.ar/",
        "https://geo.inta.gob.ar/",
        "https://geo.inta.gob.ar/api/",
        "https://geonode.senasa.gob.ar/geoserver/wms",
        "https://geonode.senasa.gob.ar/services/",
        "https://datos.gob.ar/api/3/",
        "https://datos.gob.ar/dataset?organization=agroindustria",
        "https://sisinta.inta.gob.ar/api/",
        "https://suelos.inta.gob.ar/api/",
        "https://ide.ign.gob.ar/geoserver/wms",
        "https://wms.ign.gob.ar/geoserver/wms",
        "https://geoserver.eeasalta.inta.gob.ar/geoserver/wms",
        "https://geoserver.parana.inta.gob.ar/geoserver/wms",
        "https://geointa.inta.gob.ar/geoserver/wfs",
        "https://geointa.inta.gob.ar/geoserver/wms?service=WMS&request=GetCapabilities",
        "https://datos.gob.ar/api/3/action/package_search?q=suelos"
    ]
    
    for url_base in servicios_wms:
        try:
            # Solicitar capacidades del servicio WMS
            params = {
                'service': 'WMS',
                'request': 'GetCapabilities',
                'version': '1.3.0'
            }
            
            print(f"Probando servicio WMS: {url_base}")
            response = requests.get(url_base, params=params, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Servicio WMS activo: {url_base}")
                
                # Parsear XML de capacidades
                try:
                    root = ET.fromstring(response.content)
                    
                    # Buscar capas relacionadas con suelo
                    capas_suelo = []
                    for layer in root.iter():
                        if 'Layer' in layer.tag:
                            name_elem = layer.find('.//{http://www.opengis.net/wms}Name')
                            title_elem = layer.find('.//{http://www.opengis.net/wms}Title')
                            
                            if name_elem is not None and title_elem is not None:
                                name = name_elem.text
                                title = title_elem.text
                                
                                # Buscar capas relacionadas con suelo
                                if any(keyword in title.lower() for keyword in ['suelo', 'soil', 'ph', 'materia', 'organic', 'texture', 'edafo']):
                                    capas_suelo.append({
                                        'nombre': name,
                                        'titulo': title,
                                        'servicio': url_base
                                    })
                    
                    if capas_suelo:
                        print(f"Capas de suelo encontradas: {len(capas_suelo)}")
                        return url_base, capas_suelo
                    
                except ET.ParseError as e:
                    print(f"Error parseando XML: {e}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            print(f"Error conectando a {url_base}: {e}")
            continue
    
    return None, []

def obtener_datos_suelo_wfs(lat, lon, buffer_km=5):
    """
    Obtiene datos de suelo usando servicios WFS (Web Feature Service) del INTA
    """
    # URLs de servicios WFS conocidos
    servicios_wfs = [
        "https://geointa.inta.gob.ar/geoserver/wfs",
        "https://geo.inta.gob.ar/geoserver/wfs", 
        "https://ideg.inta.gob.ar/geoserver/wfs"
    ]
    
    # Convertir buffer de km a grados (aproximado)
    buffer_grados = buffer_km / 111.0  # 1 grado ≈ 111 km
    
    # Crear bbox (bounding box) alrededor del punto
    bbox = f"{lon - buffer_grados},{lat - buffer_grados},{lon + buffer_grados},{lat + buffer_grados}"
    
    datos_suelo = []
    
    for url_wfs in servicios_wfs:
        try:
            print(f"Consultando datos de suelo en WFS: {url_wfs}")
            
            # Obtener capacidades del WFS
            params_cap = {
                'service': 'WFS',
                'request': 'GetCapabilities',
                'version': '2.0.0'
            }
            
            response_cap = requests.get(url_wfs, params=params_cap, timeout=30)
            
            if response_cap.status_code == 200:
                # Parsear capabilities para encontrar tipos de feature de suelo
                root = ET.fromstring(response_cap.content)
                
                feature_types = []
                for ft in root.iter():
                    if 'FeatureType' in ft.tag:
                        name_elem = ft.find('.//{http://www.opengis.net/wfs/2.0}Name')
                        title_elem = ft.find('.//{http://www.opengis.net/wfs/2.0}Title')
                        
                        if name_elem is not None:
                            name = name_elem.text
                            title = title_elem.text if title_elem is not None else name
                            
                            # Buscar capas de suelo
                            if any(keyword in name.lower() or keyword in title.lower() 
                                  for keyword in ['suelo', 'soil', 'edafo', 'pedologico']):
                                feature_types.append(name)
                
                print(f"Tipos de feature de suelo encontrados: {feature_types}")
                
                # Consultar cada tipo de feature de suelo
                for feature_type in feature_types[:3]:  # Limitar a los primeros 3
                    try:
                        params_feature = {
                            'service': 'WFS',
                            'request': 'GetFeature',
                            'version': '2.0.0',
                            'typeNames': feature_type,
                            'bbox': bbox,
                            'outputFormat': 'application/json',
                            'maxFeatures': 50
                        }
                        
                        response_feature = requests.get(url_wfs, params=params_feature, timeout=30)
                        
                        if response_feature.status_code == 200:
                            try:
                                geojson_data = response_feature.json()
                                
                                if 'features' in geojson_data and geojson_data['features']:
                                    print(f"✅ Datos encontrados en {feature_type}: {len(geojson_data['features'])} features")
                                    
                                    for feature in geojson_data['features']:
                                        if 'properties' in feature:
                                            propiedades = feature['properties']
                                            propiedades['tipo_capa'] = feature_type
                                            propiedades['servicio'] = url_wfs
                                            propiedades['fecha_consulta'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            datos_suelo.append(propiedades)
                                
                            except json.JSONDecodeError:
                                print(f"Respuesta no es JSON válido para {feature_type}")
                                continue
                        
                    except Exception as e:
                        print(f"Error consultando feature {feature_type}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error con servicio WFS {url_wfs}: {e}")
            continue
    
    return datos_suelo

def obtener_datos_suelo_inta_api(lat, lon):
    """
    Intenta obtener datos de suelo usando APIs específicas del INTA
    """
    # URLs de APIs conocidas del INTA
    apis_inta = [
        "https://geo.inta.gob.ar/api",
        "https://geointa.inta.gob.ar/api",
        "https://ideg.inta.gob.ar/api"
    ]
    
    datos_suelo = []
    
    for api_url in apis_inta:
        try:
            # Intentar endpoint de suelos
            endpoints = [
                f"{api_url}/suelos",
                f"{api_url}/soil",
                f"{api_url}/edafologia",
                f"{api_url}/data/suelos"
            ]
            
            for endpoint in endpoints:
                try:
                    params = {
                        'lat': lat,
                        'lon': lon,
                        'format': 'json'
                    }
                    
                    print(f"Consultando API: {endpoint}")
                    response = requests.get(endpoint, params=params, timeout=20)
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            print(f"✅ Datos encontrados en {endpoint}")
                            datos_suelo.append({
                                'fuente': endpoint,
                                'fecha_consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'latitud': lat,
                                'longitud': lon,
                                'datos': data
                            })
                        except json.JSONDecodeError:
                            print(f"Respuesta no es JSON válido en {endpoint}")
                            continue
                    else:
                        print(f"Endpoint {endpoint}: HTTP {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    print(f"Error consultando {endpoint}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error general con API {api_url}: {e}")
            continue
    
    return datos_suelo

def obtener_datos_adicionales_inta(lat, lon):
    """
    Intenta obtener datos de otros servicios del INTA que puedan tener información de suelos
    """
    # URLs adicionales del INTA
    servicios_adicionales = [
        "https://inta.gob.ar/api",
        "https://www.inta.gob.ar/api",
        "https://datos.inta.gob.ar/api",
        "https://sig.inta.gob.ar/api"
    ]
    
    datos_adicionales = []
    
    for servicio in servicios_adicionales:
        try:
            # Probar diferentes endpoints
            endpoints = [
                f"{servicio}/suelos/punto",
                f"{servicio}/geografico/suelos",
                f"{servicio}/cartas/suelos",
                f"{servicio}/relevamiento/suelos"
            ]
            
            for endpoint in endpoints:
                try:
                    params = {
                        'latitud': lat,
                        'longitud': lon,
                        'radio': 1000  # radio en metros
                    }
                    
                    print(f"Probando endpoint adicional: {endpoint}")
                    response = requests.get(endpoint, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if data:  # Si hay datos
                                print(f"✅ Datos adicionales encontrados en {endpoint}")
                                datos_adicionales.append({
                                    'fuente': endpoint,
                                    'fecha_consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'coordenadas': {'lat': lat, 'lon': lon},
                                    'datos': data
                                })
                        except json.JSONDecodeError:
                            continue
                    
                except requests.exceptions.RequestException:
                    continue
        
        except Exception:
            continue
    
    return datos_adicionales

def obtener_datos_suelo_completo(lat, lon):
    """
    Función principal que intenta obtener datos de suelo de múltiples fuentes del INTA
    """
    print(f"Obteniendo datos de suelo para coordenadas ({lat}, {lon})...")
    print("=" * 60)
    
    todos_los_datos = []
    
    # 1. Intentar servicios WMS para obtener capas disponibles
    print("\n=== CONSULTANDO SERVICIOS WMS DEL INTA ===")
    servicio_wms, capas_suelo = obtener_capacidades_wms_inta()
    
    if capas_suelo:
        print(f"Capas de suelo disponibles:")
        for capa in capas_suelo:
            print(f"  - {capa['titulo']} ({capa['nombre']})")
        
        # Agregar información de capas disponibles
        todos_los_datos.append({
            'tipo_datos': 'capas_wms_disponibles',
            'servicio': servicio_wms,
            'total_capas': len(capas_suelo),
            'capas': capas_suelo,
            'fecha_consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    else:
        print("❌ No se encontraron capas WMS de suelo")
    
    # 2. Intentar servicios WFS
    print("\n=== CONSULTANDO SERVICIOS WFS DEL INTA ===")
    datos_wfs = obtener_datos_suelo_wfs(lat, lon)
    
    if datos_wfs:
        todos_los_datos.extend(datos_wfs)
        print(f"✅ Datos WFS obtenidos: {len(datos_wfs)} registros")
    else:
        print("❌ No se encontraron datos de suelo en servicios WFS")
    
    # 3. Intentar APIs específicas
    print("\n=== CONSULTANDO APIs ESPECÍFICAS DEL INTA ===")
    datos_api = obtener_datos_suelo_inta_api(lat, lon)
    
    if datos_api:
        todos_los_datos.extend(datos_api)
        print(f"✅ Datos API obtenidos: {len(datos_api)} registros")
    else:
        print("❌ No se encontraron datos en APIs específicas del INTA")
    
    # 4. Intentar servicios adicionales
    print("\n=== CONSULTANDO SERVICIOS ADICIONALES DEL INTA ===")
    datos_adicionales = obtener_datos_adicionales_inta(lat, lon)
    
    if datos_adicionales:
        todos_los_datos.extend(datos_adicionales)
        print(f"✅ Datos adicionales obtenidos: {len(datos_adicionales)} registros")
    else:
        print("❌ No se encontraron datos en servicios adicionales")
    
    return todos_los_datos

def mostrar_resumen_datos(datos_suelo):
    """
    Muestra un resumen de los datos obtenidos
    """
    if not datos_suelo:
        print("\n" + "=" * 60)
        print("❌ NO SE PUDIERON OBTENER DATOS DE SUELO")
        print("=" * 60)
        print("Las posibles causas pueden ser:")
        print("• Las coordenadas están fuera del área de cobertura del INTA")
        print("• Los servicios del INTA no están disponibles temporalmente")
        print("• No hay datos de suelo disponibles para esa ubicación específica")
        print("• Problemas de conectividad de red")
        print("\nRecomendaciones:")
        print("1. Verificar que las coordenadas sean válidas para Argentina")
        print("2. Intentar nuevamente más tarde")
        print("3. Contactar al INTA para verificar disponibilidad de datos")
        return
    
    print(f"\n" + "=" * 60)
    print(f"✅ RESUMEN DE DATOS OBTENIDOS")
    print("=" * 60)
    print(f"Total de registros encontrados: {len(datos_suelo)}")
    
    # Contar por tipo de fuente
    tipos_fuente = {}
    for dato in datos_suelo:
        if 'tipo_capa' in dato:
            tipo = 'WFS - ' + dato['tipo_capa']
        elif 'fuente' in dato:
            tipo = 'API - ' + dato['fuente'].split('/')[-1]
        elif 'tipo_datos' in dato:
            tipo = dato['tipo_datos']
        else:
            tipo = 'Desconocido'
        
        tipos_fuente[tipo] = tipos_fuente.get(tipo, 0) + 1
    
    print("\nDatos por fuente:")
    for tipo, cantidad in tipos_fuente.items():
        print(f"  • {tipo}: {cantidad} registros")
    
    # Mostrar algunas propiedades encontradas
    propiedades_encontradas = set()
    for dato in datos_suelo:
        if isinstance(dato, dict):
            propiedades_encontradas.update(dato.keys())
    
    if propiedades_encontradas:
        print(f"\nPropiedades de datos encontradas: {len(propiedades_encontradas)}")
        propiedades_relevantes = [p for p in propiedades_encontradas 
                                if any(keyword in p.lower() for keyword in 
                                      ['ph', 'materia', 'organic', 'texture', 'clay', 'sand', 'silt', 
                                       'nutrient', 'nitrogen', 'phosphor', 'potassium'])]
        if propiedades_relevantes:
            print("Propiedades relevantes de suelo encontradas:")
            for prop in sorted(propiedades_relevantes)[:10]:  # Mostrar hasta 10
                print(f"  • {prop}")

# Ejemplo de uso
if __name__ == "__main__":
    # Coordenadas de ejemplo (Córdoba, Argentina)
    latitud = -31.4
    longitud = -64.2
    
    print("SISTEMA DE RECUPERACIÓN DE DATOS DE SUELO - INTA")
    print("Versión: Solo datos reales (sin generación artificial)")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Obtener datos de suelo
    datos_suelo = obtener_datos_suelo_completo(latitud, longitud)
    
    # Mostrar resumen
    mostrar_resumen_datos(datos_suelo)
    
    if datos_suelo:
        # Convertir a DataFrame si es posible
        try:
            # Aplanar datos complejos para el DataFrame
            datos_para_df = []
            for dato in datos_suelo:
                if isinstance(dato, dict):
                    # Si hay datos anidados, los convertimos a string
                    dato_plano = {}
                    for key, value in dato.items():
                        if isinstance(value, (dict, list)):
                            dato_plano[key] = str(value)
                        else:
                            dato_plano[key] = value
                    datos_para_df.append(dato_plano)
            
            if datos_para_df:
                df_suelo = pd.DataFrame(datos_para_df)
                
                # Guardar en CSV
                output_path = f"datos_suelo_inta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_suelo.to_csv(output_path, index=False, encoding='utf-8')
                
                print(f"\n✅ Datos guardados en: {output_path}")
                print(f"Columnas del archivo: {list(df_suelo.columns)}")
                
                if len(df_suelo) > 0:
                    print(f"\nPrimeros registros:")
                    print(df_suelo.head())
        
        except Exception as e:
            print(f"\n⚠️ Error al crear DataFrame: {e}")
            print("Los datos se mantienen en formato original (diccionarios Python)")
    
    else:
        print(f"\n⚠️ No se encontraron datos de suelo para las coordenadas ({latitud}, {longitud})")
        print("El sistema NO genera datos artificiales.")
        print("Solo retorna información real obtenida de los servicios del INTA.")