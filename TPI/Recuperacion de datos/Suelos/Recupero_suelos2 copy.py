import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

class CaracteristicasSuelo:
    """
    Clase para obtener características específicas del suelo por coordenadas
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml, application/json, text/html, */*',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        # Servicios verificados que funcionan - ACTUALIZADOS 2025
        self.servicios = {
            'ign_wms': 'https://wms.ign.gob.ar/geoserver/wms',
            'ign_wfs': 'https://wms.ign.gob.ar/geoserver/wfs',
            'geointa_wms': 'https://geointa.inta.gob.ar/geoserver/wms',
            'geointa_wfs': 'https://geointa.inta.gob.ar/geoserver/wfs',
            # Nuevas fuentes oficiales encontradas
            'indec_wms': 'https://geoservicios.indec.gob.ar/geoserver/wms',
            'indec_wfs': 'https://geoservicios.indec.gob.ar/geoserver/wfs',
            'energia_wms': 'https://sig.energia.gob.ar/wmsenergia',
            'energia_wfs': 'https://sig.energia.gob.ar/wfsenergia',
            'conae_wms': 'https://geoservicios.conae.gov.ar/geoserver/wms',
            'conae_wfs': 'https://geoservicios.conae.gov.ar/geoserver/wfs'
        }
        
        # Capas de suelo conocidas del IGN (encontradas anteriormente)
        self.capas_suelo_ign = [
            'ign:edafologia_afloramiento_rocoso',
            'ign:edafologia_suelos',
            'ign:edafologia_capacidad_uso',
            'ign:edafologia_erosion',
            'ign:edafologia_drenaje',
            'ign:edafologia_ph',
            'ign:edafologia_textura'
        ]
    
    def obtener_capas_disponibles(self, servicio_wms):
        """Obtiene todas las capas disponibles de un servicio WMS"""
        try:
            params = {
                'SERVICE': 'WMS',
                'REQUEST': 'GetCapabilities',
                'VERSION': '1.1.1'
            }
            
            response = requests.get(servicio_wms, params=params, headers=self.headers, timeout=30, verify=False)
            
            if response.status_code == 200:
                try:
                    root = ET.fromstring(response.content)
                    capas_encontradas = []
                    
                    for layer in root.iter():
                        if layer.tag.endswith('Layer') or layer.tag == 'Layer':
                            name_elem = None
                            title_elem = None
                            
                            for child in layer:
                                if child.tag.endswith('Name') or child.tag == 'Name':
                                    name_elem = child
                                elif child.tag.endswith('Title') or child.tag == 'Title':
                                    title_elem = child
                            
                            if name_elem is not None and name_elem.text:
                                name = name_elem.text.strip()
                                title = title_elem.text.strip() if title_elem is not None and title_elem.text else name
                                
                                # Buscar capas relacionadas con suelo - AMPLIADAS
                                keywords_suelo = [
                                    'suelo', 'soil', 'edafo', 'ph', 'textura', 'texture',
                                    'erosion', 'drenaje', 'drainage', 'capacidad', 'uso',
                                    'aptitud', 'fertilidad', 'materia', 'organic', 'carbon',
                                    'agr', 'farm', 'crop', 'cultivo', 'agricultura', 'rural',
                                    'land', 'tierra', 'productivity', 'productividad', 'classification',
                                    'tipo', 'type', 'serie', 'series', 'taxonomy', 'taxonomia'
                                ]
                                
                                if any(keyword in name.lower() or keyword in title.lower() 
                                      for keyword in keywords_suelo):
                                    capas_encontradas.append({
                                        'nombre': name,
                                        'titulo': title
                                    })
                    
                    return capas_encontradas
                except ET.ParseError:
                    return []
            return []
        except Exception as e:
            print(f"Error obteniendo capas: {e}")
            return []
    
    def consultar_punto_wms(self, servicio_wms, capa, lat, lon):
        """Consulta información de un punto específico usando GetFeatureInfo"""
        try:
            # Parámetros para GetFeatureInfo
            params = {
                'SERVICE': 'WMS',
                'VERSION': '1.1.1',
                'REQUEST': 'GetFeatureInfo',
                'LAYERS': capa,
                'QUERY_LAYERS': capa,
                'SRS': 'EPSG:4326',
                'BBOX': f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",
                'WIDTH': 101,
                'HEIGHT': 101,
                'X': 50,
                'Y': 50,
                'INFO_FORMAT': 'text/plain',
                'FEATURE_COUNT': 1
            }
            
            response = requests.get(servicio_wms, params=params, headers=self.headers, timeout=20, verify=False)
            
            if response.status_code == 200 and response.text.strip():
                return response.text.strip()
            return None
            
        except Exception as e:
            return None
    
    def consultar_area_wfs(self, servicio_wfs, capa, lat, lon, radio_km=20):
        """Consulta datos de un área usando WFS"""
        try:
            # Convertir radio a grados aproximados
            radio_grados = radio_km / 111.0
            
            bbox = f"{lon-radio_grados},{lat-radio_grados},{lon+radio_grados},{lat+radio_grados}"
            
            params = {
                'SERVICE': 'WFS',
                'VERSION': '1.1.0',
                'REQUEST': 'GetFeature',
                'TYPENAME': capa,
                'BBOX': bbox,
                'OUTPUTFORMAT': 'application/json',
                'MAXFEATURES': 10
            }
            
            response = requests.get(servicio_wfs, params=params, headers=self.headers, timeout=20, verify=False)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('features'):
                        return data['features']
                except json.JSONDecodeError:
                    pass
            return []
            
        except Exception as e:
            return []
    
    def calcular_distancia(self, lat1, lon1, lat2, lon2):
        """Calcula distancia entre dos puntos en km"""
        R = 6371  # Radio de la Tierra en km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def obtener_caracteristicas_suelo(self, lat, lon):
        """
        Función principal que obtiene características del suelo para coordenadas específicas
        """
        print(f"🎯 OBTENIENDO CARACTERÍSTICAS DEL SUELO")
        print(f"📍 Coordenadas: ({lat}, {lon})")
        print("="*60)
        
        caracteristicas = {
            'coordenadas': {'latitud': lat, 'longitud': lon},
            'fecha_consulta': datetime.now().isoformat(),
            'datos_encontrados': [],
            'servicios_consultados': []
        }
        
        # 1. Consultar IGN (servicio que sabemos funciona)
        print("\n🗺️ Consultando IGN (Instituto Geográfico Nacional)...")
        
        # Obtener capas disponibles del IGN
        capas_ign = self.obtener_capas_disponibles(self.servicios['ign_wms'])
        if capas_ign:
            print(f"✅ Encontradas {len(capas_ign)} capas de suelo en IGN")
            
            for capa in capas_ign[:5]:  # Limitar a 5 capas para no sobrecargar
                nombre_capa = capa['nombre']
                titulo_capa = capa['titulo']
                
                print(f"🔍 Consultando: {titulo_capa}")
                
                # Consultar punto específico
                info_punto = self.consultar_punto_wms(self.servicios['ign_wms'], nombre_capa, lat, lon)
                
                if info_punto and len(info_punto) > 10 and "no features were found" not in info_punto.lower():
                    caracteristica = {
                        'fuente': 'IGN',
                        'capa': nombre_capa,
                        'titulo': titulo_capa,
                        'tipo_consulta': 'punto_exacto',
                        'informacion': info_punto,
                        'distancia_km': 0
                    }
                    caracteristicas['datos_encontrados'].append(caracteristica)
                    print(f"  ✅ Datos encontrados: {info_punto[:100]}...")
                elif info_punto:
                    print(f"  ⚠️ Sin datos en punto exacto: {info_punto}")
                
                # También consultar área cercana via WFS con múltiples radios
                for radio in [5, 15, 30]:  # Probar con radios crecientes
                    datos_area = self.consultar_area_wfs(self.servicios['ign_wfs'], nombre_capa, lat, lon, radio)
                    
                    if datos_area:
                        print(f"  ✅ Datos encontrados en radio {radio}km")
                        
                        for feature in datos_area:
                            if 'properties' in feature and feature['properties']:
                                props = feature['properties']
                                
                                # Verificar que hay propiedades con contenido real
                                props_con_contenido = {k: v for k, v in props.items() 
                                                     if v and str(v).strip() and str(v) != 'null'}
                                
                                if props_con_contenido:
                                    # Calcular distancia si hay geometría
                                    distancia = radio  # Usar radio como aproximación
                                    if 'geometry' in feature and feature['geometry'].get('coordinates'):
                                        coords = feature['geometry']['coordinates']
                                        if isinstance(coords[0], list):  # Polígono
                                            punto_geom = coords[0][0] if isinstance(coords[0][0], list) else coords[0]
                                        else:  # Punto
                                            punto_geom = coords
                                        
                                        if len(punto_geom) >= 2:
                                            distancia = self.calcular_distancia(lat, lon, punto_geom[1], punto_geom[0])
                                    
                                    caracteristica = {
                                        'fuente': 'IGN',
                                        'capa': nombre_capa,
                                        'titulo': titulo_capa,
                                        'tipo_consulta': f'area_cercana_radio_{radio}km',
                                        'propiedades': props_con_contenido,
                                        'distancia_km': round(distancia, 2)
                                    }
                                    caracteristicas['datos_encontrados'].append(caracteristica)
                                    print(f"    • Propiedades: {len(props_con_contenido)}, distancia: {distancia:.1f}km")
                        break  # Si encontramos datos, no probar radios mayores
        
        caracteristicas['servicios_consultados'].append('IGN')
        
        # 2. Consultar INDEC (nuevos geoservicios oficiales)
        print(f"\n📊 Consultando INDEC (Instituto Nacional de Estadística y Censos)...")
        capas_indec = self.obtener_capas_disponibles(self.servicios.get('indec_wms'))
        
        if capas_indec:
            print(f"✅ Encontradas {len(capas_indec)} capas de suelo en INDEC")
            
            for capa in capas_indec[:3]:  # Limitar a 3 capas
                nombre_capa = capa['nombre']
                titulo_capa = capa['titulo']
                
                print(f"🔍 Consultando INDEC: {titulo_capa}")
                
                # Consultar punto específico
                info_punto = self.consultar_punto_wms(self.servicios['indec_wms'], nombre_capa, lat, lon)
                
                if info_punto and len(info_punto) > 10 and "no features were found" not in info_punto.lower():
                    caracteristica = {
                        'fuente': 'INDEC',
                        'capa': nombre_capa,
                        'titulo': titulo_capa,
                        'tipo_consulta': 'punto_exacto',
                        'informacion': info_punto,
                        'distancia_km': 0
                    }
                    caracteristicas['datos_encontrados'].append(caracteristica)
                    print(f"  ✅ Datos INDEC encontrados: {info_punto[:100]}...")
        else:
            print("⚠️ INDEC no disponible o sin capas de suelo")
        
        caracteristicas['servicios_consultados'].append('INDEC')
        
        # 3. Consultar INTA (si está disponible)
        print(f"\n🌾 Consultando INTA...")
        capas_inta = self.obtener_capas_disponibles(self.servicios['geointa_wms'])
        
        if capas_inta:
            print(f"✅ Encontradas {len(capas_inta)} capas de suelo en INTA")
            
            for capa in capas_inta[:3]:  # Limitar a 3 capas
                nombre_capa = capa['nombre']
                titulo_capa = capa['titulo']
                
                print(f"🔍 Consultando INTA: {titulo_capa}")
                
                # Consultar área cercana
                datos_area = self.consultar_area_wfs(self.servicios['geointa_wfs'], nombre_capa, lat, lon)
                
                if datos_area:
                    for feature in datos_area:
                        if 'properties' in feature:
                            caracteristica = {
                                'fuente': 'INTA',
                                'capa': nombre_capa,
                                'titulo': titulo_capa,
                                'tipo_consulta': 'area_cercana',
                                'propiedades': feature['properties'],
                                'distancia_km': 'calculando...'
                            }
                            caracteristicas['datos_encontrados'].append(caracteristica)
                            print(f"  ✅ Datos INTA encontrados")
        else:
            print("⚠️ INTA no disponible temporalmente")
        
        caracteristicas['servicios_consultados'].append('INTA')
        
        # 4. Consultar CONAE (Comisión Nacional de Actividades Espaciales)
        print(f"\n🛰️ Consultando CONAE...")
        capas_conae = self.obtener_capas_disponibles(self.servicios.get('conae_wms'))
        
        if capas_conae:
            print(f"✅ Encontradas {len(capas_conae)} capas de suelo en CONAE")
            
            for capa in capas_conae[:2]:  # Limitar a 2 capas
                nombre_capa = capa['nombre']
                titulo_capa = capa['titulo']
                
                print(f"🔍 Consultando CONAE: {titulo_capa}")
                
                # Consultar área cercana
                datos_area = self.consultar_area_wfs(self.servicios['conae_wfs'], nombre_capa, lat, lon)
                
                if datos_area:
                    for feature in datos_area:
                        if 'properties' in feature and feature['properties']:
                            caracteristica = {
                                'fuente': 'CONAE',
                                'capa': nombre_capa,
                                'titulo': titulo_capa,
                                'tipo_consulta': 'area_cercana',
                                'propiedades': feature['properties'],
                                'distancia_km': 'calculando...'
                            }
                            caracteristicas['datos_encontrados'].append(caracteristica)
                            print(f"  ✅ Datos CONAE encontrados")
        else:
            print("⚠️ CONAE no disponible o sin capas de suelo")
        
        caracteristicas['servicios_consultados'].append('CONAE')
        
        return caracteristicas
    
    def generar_reporte_caracteristicas(self, caracteristicas):
        """Genera un reporte legible de las características encontradas"""
        
        datos = caracteristicas['datos_encontrados']
        
        if not datos:
            print(f"\n❌ NO SE ENCONTRARON CARACTERÍSTICAS DE SUELO")
            print(f"📍 Para las coordenadas: {caracteristicas['coordenadas']['latitud']}, {caracteristicas['coordenadas']['longitud']}")
            print(f"\n💡 SUGERENCIAS:")
            print(f"• Probar con coordenadas de zonas agrícolas conocidas")
            print(f"• Aumentar el radio de búsqueda")
            print(f"• Verificar que las coordenadas sean de Argentina")
            return None
        
        print(f"\n🎉 CARACTERÍSTICAS DE SUELO ENCONTRADAS")
        print("="*60)
        print(f"📍 Coordenadas: {caracteristicas['coordenadas']['latitud']}, {caracteristicas['coordenadas']['longitud']}")
        print(f"📊 Total de datos: {len(datos)}")
        print(f"🕐 Fecha consulta: {caracteristicas['fecha_consulta'][:19]}")
        
        # Agrupar por fuente
        por_fuente = {}
        for dato in datos:
            fuente = dato['fuente']
            if fuente not in por_fuente:
                por_fuente[fuente] = []
            por_fuente[fuente].append(dato)
        
        for fuente, datos_fuente in por_fuente.items():
            print(f"\n🗂️ === DATOS DE {fuente} ===")
            
            for i, dato in enumerate(datos_fuente, 1):
                print(f"\n{i}. {dato['titulo']}")
                print(f"   📏 Distancia: {dato.get('distancia_km', 'N/A')} km")
                print(f"   🔍 Tipo: {dato['tipo_consulta']}")
                
                # Mostrar información específica
                if 'informacion' in dato:
                    info = dato['informacion']
                    if len(info) > 200:
                        info = info[:200] + "..."
                    print(f"   📋 Info: {info}")
                
                if 'propiedades' in dato:
                    props = dato['propiedades']
                    print(f"   📊 Propiedades encontradas: {len(props)}")
                    
                    # Mostrar propiedades más relevantes
                    props_relevantes = {}
                    for key, value in props.items():
                        if value and str(value).strip() and str(value) != 'null':
                            # Filtrar propiedades relevantes para suelos
                            key_lower = key.lower()
                            if any(keyword in key_lower for keyword in 
                                  ['ph', 'textura', 'texture', 'clay', 'sand', 'silt', 
                                   'organic', 'carbon', 'nitrogen', 'phosphor', 'potassium',
                                   'drainage', 'erosion', 'depth', 'capacity', 'fertility']):
                                props_relevantes[key] = value
                    
                    if props_relevantes:
                        print(f"   🎯 Propiedades de suelo:")
                        for key, value in list(props_relevantes.items())[:5]:  # Mostrar hasta 5
                            valor_mostrar = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                            print(f"      • {key}: {valor_mostrar}")
                    else:
                        # Mostrar algunas propiedades generales
                        props_generales = list(props.items())[:3]
                        if props_generales:
                            print(f"   📝 Propiedades generales:")
                            for key, value in props_generales:
                                if value and str(value).strip():
                                    valor_mostrar = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                                    print(f"      • {key}: {valor_mostrar}")
        
        return caracteristicas
    
    def guardar_caracteristicas(self, caracteristicas, lat, lon):
        """Guarda las características en archivos"""
        if not caracteristicas['datos_encontrados']:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"caracteristicas_suelo_{lat}_{lon}_{timestamp}"
        
        # Guardar JSON completo
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(caracteristicas, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Características guardadas en: {json_file}")
        
        # Crear resumen en CSV
        try:
            datos_csv = []
            for dato in caracteristicas['datos_encontrados']:
                fila = {
                    'fuente': dato['fuente'],
                    'capa': dato['titulo'],
                    'tipo_consulta': dato['tipo_consulta'],
                    'distancia_km': dato.get('distancia_km', ''),
                    'tiene_informacion': 'Sí' if 'informacion' in dato else 'No',
                    'num_propiedades': len(dato.get('propiedades', {})),
                    'coordenadas': f"{lat}, {lon}"
                }
                datos_csv.append(fila)
            
            df = pd.DataFrame(datos_csv)
            csv_file = f"{base_filename}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"📊 Resumen guardado en: {csv_file}")
            
            return json_file, csv_file
            
        except Exception as e:
            print(f"⚠️ Error creando CSV: {e}")
            return json_file, None

def obtener_caracteristicas_suelo_por_coordenadas(latitud, longitud):
    """
    Función principal para obtener características del suelo por coordenadas
    
    Args:
        latitud (float): Latitud en grados decimales
        longitud (float): Longitud en grados decimales
    
    Returns:
        dict: Diccionario con las características del suelo encontradas
    """
    
    # Validar coordenadas básicas
    if not (-90 <= latitud <= 90) or not (-180 <= longitud <= 180):
        print("❌ Error: Coordenadas inválidas")
        print(f"   Latitud debe estar entre -90 y 90 (recibida: {latitud})")
        print(f"   Longitud debe estar entre -180 y 180 (recibida: {longitud})")
        return None
    
    # Advertir si las coordenadas no parecen ser de Argentina
    if not (-55 <= latitud <= -21.8 and -73.6 <= longitud <= -53.6):
        print("⚠️ ADVERTENCIA: Las coordenadas no parecen estar en Argentina")
        print("   Este sistema está optimizado para suelos argentinos")
        respuesta = input("¿Continuar de todas formas? (s/n): ").lower()
        if respuesta != 's':
            return None
    
    # Crear instancia y obtener características
    extractor = CaracteristicasSuelo()
    caracteristicas = extractor.obtener_caracteristicas_suelo(latitud, longitud)
    
    # Generar reporte
    extractor.generar_reporte_caracteristicas(caracteristicas)
    
    # Guardar archivos
    if caracteristicas['datos_encontrados']:
        extractor.guardar_caracteristicas(caracteristicas, latitud, longitud)
    
    return caracteristicas

# Ejemplo de uso
if __name__ == "__main__":
    print("🌍 SISTEMA DE CARACTERÍSTICAS DE SUELO POR COORDENADAS")
    print("📅 Versión 2025 - Solo datos reales de fuentes oficiales")
    print("="*70)
    
    # Coordenadas de ejemplo - ACTUALIZADO con zonas que tienen más datos
    # Probemos varias coordenadas para encontrar datos reales
    
    coordenadas_prueba = [
        (-34.6037, -58.3816, "Buenos Aires Capital"),
        (-31.4201, -64.1888, "Córdoba Capital"),  # Más específica de Córdoba ciudad
        (-32.9442, -60.6505, "Rosario, Santa Fe"),  # Zona agrícola importante
        (-31.7333, -60.5333, "Paraná, Entre Ríos"),  # Zona con estudios de suelo
        (-34.9215, -57.9545, "La Plata, Buenos Aires"),  # Zona universitaria con estudios
    ]
    
    print(f"🔍 PROBANDO MÚLTIPLES COORDENADAS PARA ENCONTRAR DATOS REALES")
    
    datos_encontrados = False
    
    for lat, lon, descripcion in coordenadas_prueba:
        print(f"\n📍 Probando: {descripcion} ({lat}, {lon})")
        resultado = obtener_caracteristicas_suelo_por_coordenadas(lat, lon)
        
        if resultado and resultado['datos_encontrados']:
            # Verificar si hay datos reales (no solo "no features were found")
            datos_reales = [d for d in resultado['datos_encontrados'] 
                          if d.get('informacion', '') != 'no features were found']
            
            if datos_reales:
                print(f"✅ ¡DATOS ENCONTRADOS EN {descripcion}!")
                datos_encontrados = True
                break
            else:
                print(f"⚠️ Sin datos específicos en {descripcion}")
        else:
            print(f"❌ Sin respuesta para {descripcion}")
    
    # Si no se encontraron datos en las coordenadas de prueba, usar coordenadas por defecto
    if not datos_encontrados:
        print(f"\n🎯 EJEMPLO: Consultando características del suelo")
        LATITUD = -31.4201  # Córdoba Capital
        LONGITUD = -64.1888
        print(f"📍 Coordenadas: {LATITUD}, {LONGITUD} (Córdoba, Argentina)")
        
        # Obtener características
        resultado = obtener_caracteristicas_suelo_por_coordenadas(LATITUD, LONGITUD)
        
        if resultado and resultado['datos_encontrados']:
            print(f"\n✅ PROCESO COMPLETADO EXITOSAMENTE")
            print(f"📊 Se encontraron {len(resultado['datos_encontrados'])} registros de características de suelo")
            print(f"🗂️ Archivos generados con toda la información detallada")
        else:
            print(f"\n⚠️ No se pudieron obtener características específicas para esta ubicación")
            print(f"💡 Intenta con coordenadas de zonas agrícolas conocidas:")
            print(f"   • Buenos Aires: -34.6, -58.4")
            print(f"   • Santa Fe: -31.6, -60.7") 
            print(f"   • Entre Ríos: -31.7, -60.5")
    
    print(f"\n🏁 Proceso finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")