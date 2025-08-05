import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import time
from urllib.parse import urljoin, quote
import warnings
warnings.filterwarnings('ignore')

class RecuperadorSuelosINTA:
    """
    Clase para recuperar datos de suelos de diversas fuentes argentinas
    """
    
    def __init__(self):
        # URLs verificadas y actualizadas 2025
        self.servicios_activos = {
            'geointa_wms': 'https://geointa.inta.gob.ar/geoserver/wms',
            'geointa_wfs': 'https://geointa.inta.gob.ar/geoserver/wfs',
            'geo_inta': 'https://geo.inta.gob.ar',
            'senasa_wms': 'https://geonode.senasa.gob.ar/geoserver/wms',
            'senasa_wfs': 'https://geonode.senasa.gob.ar/geoserver/wfs',
            'ign_wms': 'https://wms.ign.gob.ar/geoserver/wms',
            'ign_wfs': 'https://wms.ign.gob.ar/geoserver/wfs',
            'datos_gob': 'https://datos.gob.ar/api/3',
            'visor_geointa': 'https://visor.geointa.inta.gob.ar'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/xml, application/json, text/html, */*',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        self.datos_obtenidos = []
    
    def probar_conexion_servicio(self, url, timeout=10):
        """Prueba si un servicio está disponible"""
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout, verify=False)
            return response.status_code == 200, response
        except Exception as e:
            return False, str(e)
    
    def obtener_capacidades_wms(self, url_wms):
        """Obtiene las capacidades de un servicio WMS correctamente"""
        try:
            # Parámetros correctos para GetCapabilities
            params = {
                'SERVICE': 'WMS',
                'REQUEST': 'GetCapabilities',
                'VERSION': '1.1.1'  # Usar versión más compatible
            }
            
            print(f"📡 Consultando capacidades WMS: {url_wms}")
            response = requests.get(url_wms, params=params, headers=self.headers, timeout=30, verify=False)
            
            if response.status_code == 200:
                # Verificar si el contenido es XML
                content_type = response.headers.get('content-type', '').lower()
                if 'xml' in content_type or response.text.strip().startswith('<?xml'):
                    try:
                        # Limpiar el XML antes de parsearlo
                        xml_content = response.text.encode('utf-8')
                        root = ET.fromstring(xml_content)
                        
                        # Buscar capas con diferentes namespaces
                        capas_suelo = []
                        namespaces = {
                            'wms': 'http://www.opengis.net/wms',
                            '': ''  # namespace vacío
                        }
                        
                        # Buscar todos los elementos Layer
                        for layer in root.iter():
                            if layer.tag.endswith('Layer') or layer.tag == 'Layer':
                                name_elem = None
                                title_elem = None
                                
                                # Buscar elementos Name y Title
                                for child in layer:
                                    if child.tag.endswith('Name') or child.tag == 'Name':
                                        name_elem = child
                                    elif child.tag.endswith('Title') or child.tag == 'Title':
                                        title_elem = child
                                
                                if name_elem is not None and name_elem.text:
                                    name = name_elem.text.strip()
                                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else name
                                    
                                    # Buscar capas relacionadas con suelo (palabras clave más amplias)
                                    keywords_suelo = [
                                        'suelo', 'soil', 'ph', 'materia', 'organic', 'texture', 
                                        'edafo', 'pedologic', 'clay', 'sand', 'silt', 'carbon',
                                        'nutrient', 'fertility', 'erosion', 'degradation',
                                        'capacidad', 'aptitud', 'uso', 'agricultura'
                                    ]
                                    
                                    if any(keyword in name.lower() or keyword in title.lower() 
                                          for keyword in keywords_suelo):
                                        capas_suelo.append({
                                            'nombre': name,
                                            'titulo': title,
                                            'servicio': url_wms
                                        })
                        
                        if capas_suelo:
                            print(f"✅ Encontradas {len(capas_suelo)} capas de suelo")
                            return capas_suelo
                        else:
                            print(f"⚠️ Servicio activo pero sin capas de suelo específicas")
                            return []
                            
                    except ET.ParseError as e:
                        print(f"❌ Error parseando XML: {e}")
                        # Intentar con un enfoque más simple
                        if 'suelo' in response.text.lower() or 'soil' in response.text.lower():
                            print("📄 Se detectó contenido relacionado con suelos en la respuesta")
                            return [{'detalle': 'Contenido de suelos detectado pero no parseado', 'servicio': url_wms}]
                        return []
                else:
                    print(f"❌ Respuesta no es XML válido: {content_type}")
                    return []
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error consultando WMS {url_wms}: {e}")
            return []
    
    def obtener_datos_wfs_mejorado(self, url_wfs, lat, lon, buffer_km=10):
        """Consulta mejorada de datos WFS"""
        try:
            # Obtener capacidades primero
            params_cap = {
                'SERVICE': 'WFS',
                'REQUEST': 'GetCapabilities',
                'VERSION': '1.1.0'
            }
            
            print(f"📡 Consultando WFS: {url_wfs}")
            response_cap = requests.get(url_wfs, params=params_cap, headers=self.headers, timeout=30, verify=False)
            
            if response_cap.status_code == 200:
                try:
                    root = ET.fromstring(response_cap.content)
                    
                    # Buscar FeatureTypes
                    feature_types = []
                    for elem in root.iter():
                        if elem.tag.endswith('FeatureType') or elem.tag == 'FeatureType':
                            name_elem = None
                            title_elem = None
                            
                            for child in elem:
                                if child.tag.endswith('Name') or child.tag == 'Name':
                                    name_elem = child
                                elif child.tag.endswith('Title') or child.tag == 'Title':
                                    title_elem = child
                            
                            if name_elem is not None and name_elem.text:
                                name = name_elem.text.strip()
                                title = title_elem.text.strip() if title_elem is not None and title_elem.text else name
                                
                                # Filtrar por términos relacionados con suelo
                                if any(keyword in name.lower() or keyword in title.lower() 
                                      for keyword in ['suelo', 'soil', 'edafo', 'pedologic', 'agricultura', 'aptitud']):
                                    feature_types.append({
                                        'name': name,
                                        'title': title
                                    })
                    
                    if feature_types:
                        print(f"✅ Encontrados {len(feature_types)} tipos de feature de suelo")
                        
                        # Consultar datos para cada feature type
                        datos_encontrados = []
                        for ft in feature_types[:2]:  # Limitar a 2 para evitar timeout
                            datos_ft = self.consultar_feature_type(url_wfs, ft['name'], lat, lon, buffer_km)
                            if datos_ft:
                                datos_encontrados.extend(datos_ft)
                        
                        return datos_encontrados
                    else:
                        print("⚠️ No se encontraron feature types de suelo")
                        return []
                        
                except ET.ParseError as e:
                    print(f"❌ Error parseando capabilities: {e}")
                    return []
            else:
                print(f"❌ Error obteniendo capabilities: {response_cap.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error en consulta WFS: {e}")
            return []
    
    def consultar_feature_type(self, url_wfs, typename, lat, lon, buffer_km):
        """Consulta un feature type específico"""
        try:
            # Crear bbox
            buffer_grados = buffer_km / 111.0
            bbox = f"{lon - buffer_grados},{lat - buffer_grados},{lon + buffer_grados},{lat + buffer_grados}"
            
            params = {
                'SERVICE': 'WFS',
                'REQUEST': 'GetFeature',
                'VERSION': '1.1.0',
                'TYPENAME': typename,
                'BBOX': bbox,
                'OUTPUTFORMAT': 'GML2',  # Formato más compatible
                'MAXFEATURES': 20
            }
            
            print(f"🔍 Consultando feature: {typename}")
            response = requests.get(url_wfs, params=params, headers=self.headers, timeout=30, verify=False)
            
            if response.status_code == 200 and response.content:
                # Intentar parsear como GML
                try:
                    root = ET.fromstring(response.content)
                    features_encontrados = []
                    
                    # Buscar elementos que contengan datos
                    for elem in root.iter():
                        if elem.text and len(elem.text.strip()) > 0:
                            # Si encontramos texto significativo, es un dato
                            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                            if not tag_name.startswith('gml') and len(elem.text.strip()) > 1:
                                features_encontrados.append({
                                    'campo': tag_name,
                                    'valor': elem.text.strip(),
                                    'feature_type': typename,
                                    'servicio': url_wfs
                                })
                    
                    if features_encontrados:
                        print(f"✅ Datos encontrados en {typename}: {len(features_encontrados)} campos")
                        return features_encontrados
                    else:
                        print(f"⚠️ Feature {typename} sin datos específicos")
                        return []
                        
                except ET.ParseError:
                    # Si no es XML válido, intentar buscar patrones en texto plano
                    text_content = response.text.lower()
                    if any(keyword in text_content for keyword in ['suelo', 'ph', 'materia', 'organic']):
                        print(f"📄 Contenido de suelos detectado en {typename}")
                        return [{
                            'feature_type': typename,
                            'servicio': url_wfs,
                            'contenido': 'Datos de suelos detectados',
                            'tamano_respuesta': len(response.content)
                        }]
                    return []
            else:
                print(f"❌ Error consultando {typename}: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error en feature {typename}: {e}")
            return []
    
    def consultar_datos_gob_ar(self, terminos_busqueda=['suelos', 'agricultura', 'INTA']):
        """Consulta la API de datos.gob.ar"""
        try:
            print("📊 Consultando datos.gob.ar...")
            datos_encontrados = []
            
            for termino in terminos_busqueda:
                url = f"{self.servicios_activos['datos_gob']}/action/package_search"
                params = {
                    'q': termino,
                    'rows': 20,
                    'sort': 'metadata_modified desc'
                }
                
                response = requests.get(url, params=params, headers=self.headers, timeout=20, verify=False)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get('success') and data.get('result', {}).get('results'):
                            datasets = data['result']['results']
                            print(f"✅ Encontrados {len(datasets)} datasets para '{termino}'")
                            
                            for dataset in datasets:
                                # Filtrar datasets relevantes para suelos
                                title = dataset.get('title', '').lower()
                                notes = dataset.get('notes', '').lower()
                                tags = [tag.get('name', '') for tag in dataset.get('tags', [])]
                                
                                if any(keyword in title or keyword in notes 
                                      for keyword in ['suelo', 'agricultura', 'edaf', 'ph', 'fertilidad']):
                                    
                                    dataset_info = {
                                        'fuente': 'datos.gob.ar',
                                        'titulo': dataset.get('title'),
                                        'descripcion': dataset.get('notes', '')[:200] + '...' if len(dataset.get('notes', '')) > 200 else dataset.get('notes', ''),
                                        'url': f"https://datos.gob.ar/dataset/{dataset.get('name')}",
                                        'tags': tags,
                                        'recursos': len(dataset.get('resources', [])),
                                        'fecha_actualizacion': dataset.get('metadata_modified'),
                                        'organizacion': dataset.get('organization', {}).get('title', ''),
                                        'termino_busqueda': termino
                                    }
                                    
                                    datos_encontrados.append(dataset_info)
                        
                    except json.JSONDecodeError:
                        print(f"❌ Error parseando JSON para {termino}")
                        continue
                else:
                    print(f"❌ Error HTTP {response.status_code} para {termino}")
            
            return datos_encontrados
            
        except Exception as e:
            print(f"❌ Error consultando datos.gob.ar: {e}")
            return []
    
    def obtener_datos_suelo_completo(self, lat, lon, incluir_datasets=True):
        """Función principal mejorada"""
        print(f"🌍 RECUPERANDO DATOS DE SUELO PARA ({lat}, {lon})")
        print("=" * 70)
        
        todos_los_datos = []
        
        # 1. Probar servicios WMS
        print("\n🗺️ === CONSULTANDO SERVICIOS WMS ===")
        for nombre, url in [('GeoINTA', self.servicios_activos['geointa_wms']),
                           ('SENASA', self.servicios_activos['senasa_wms']),
                           ('IGN', self.servicios_activos['ign_wms'])]:
            
            capas = self.obtener_capacidades_wms(url)
            if capas:
                todos_los_datos.append({
                    'tipo': 'WMS_Capabilities',
                    'servicio': nombre,
                    'url': url,
                    'capas_suelo': capas,
                    'total_capas': len(capas),
                    'fecha_consulta': datetime.now().isoformat()
                })
            time.sleep(1)  # Evitar sobrecarga
        
        # 2. Probar servicios WFS
        print(f"\n📊 === CONSULTANDO SERVICIOS WFS ===")
        for nombre, url in [('GeoINTA', self.servicios_activos['geointa_wfs']),
                           ('SENASA', self.servicios_activos['senasa_wfs'])]:
            
            datos_wfs = self.obtener_datos_wfs_mejorado(url, lat, lon)
            if datos_wfs:
                todos_los_datos.extend(datos_wfs)
            time.sleep(1)
        
        # 3. Consultar datos.gob.ar si se solicita
        if incluir_datasets:
            print(f"\n📋 === CONSULTANDO DATASETS PÚBLICOS ===")
            datasets = self.consultar_datos_gob_ar()
            if datasets:
                todos_los_datos.extend(datasets)
        
        return todos_los_datos
    
    def generar_reporte(self, datos):
        """Genera un reporte detallado de los datos obtenidos"""
        if not datos:
            print("\n" + "="*70)
            print("❌ NO SE PUDIERON OBTENER DATOS DE SUELO")
            print("="*70)
            print("\n🔍 POSIBLES SOLUCIONES:")
            print("• Verificar conectividad de internet")
            print("• Probar con coordenadas de zonas agrícolas conocidas")
            print("• Contactar directamente al INTA: geo@inta.gob.ar")
            print("• Revisar el visor web: https://visor.geointa.inta.gob.ar")
            return None
        
        print(f"\n" + "="*70)
        print(f"✅ DATOS DE SUELO OBTENIDOS EXITOSAMENTE")
        print("="*70)
        print(f"📊 Total de registros: {len(datos)}")
        
        # Contar por tipo
        tipos = {}
        for dato in datos:
            if isinstance(dato, dict):
                tipo = dato.get('tipo', dato.get('fuente', 'Desconocido'))
                tipos[tipo] = tipos.get(tipo, 0) + 1
        
        print(f"\n📈 DISTRIBUCIÓN POR FUENTE:")
        for tipo, cantidad in tipos.items():
            print(f"   • {tipo}: {cantidad} registros")
        
        # Identificar servicios activos
        servicios_activos = set()
        for dato in datos:
            if isinstance(dato, dict):
                if 'servicio' in dato:
                    servicios_activos.add(dato['servicio'].split('/')[2] if '//' in dato['servicio'] else dato['servicio'])
                elif 'url' in dato:
                    servicios_activos.add(dato['url'].split('/')[2] if '//' in dato['url'] else dato['url'])
        
        if servicios_activos:
            print(f"\n🌐 SERVICIOS ACTIVOS ENCONTRADOS:")
            for servicio in sorted(servicios_activos):
                print(f"   • {servicio}")
        
        return datos
    
    def guardar_datos(self, datos, lat, lon):
        """Guarda los datos en diferentes formatos"""
        if not datos:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"suelos_argentina_{lat}_{lon}_{timestamp}"
        
        # Guardar como JSON (más completo)
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 Datos guardados en JSON: {json_file}")
        
        # Intentar crear CSV simplificado
        try:
            datos_planos = []
            for item in datos:
                if isinstance(item, dict):
                    # Aplanar diccionarios anidados
                    item_plano = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            item_plano[key] = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                        else:
                            item_plano[key] = value
                    datos_planos.append(item_plano)
            
            if datos_planos:
                df = pd.DataFrame(datos_planos)
                csv_file = f"{base_filename}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"📊 Datos guardados en CSV: {csv_file}")
                
                return json_file, csv_file
        
        except Exception as e:
            print(f"⚠️ No se pudo crear CSV: {e}")
        
        return json_file, None

# Función principal de uso
def recuperar_datos_suelo_argentina(lat, lon, incluir_datasets=True):
    """
    Función principal para recuperar datos de suelo de Argentina
    
    Args:
        lat (float): Latitud (coordenadas decimales)
        lon (float): Longitud (coordenadas decimales)
        incluir_datasets (bool): Si incluir datasets de datos.gob.ar
    
    Returns:
        list: Lista de datos obtenidos de diferentes fuentes
    """
    recuperador = RecuperadorSuelosINTA()
    
    print("🚀 SISTEMA DE RECUPERACIÓN DE DATOS DE SUELO - ARGENTINA")
    print("📅 Versión actualizada 2025")
    print(f"🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validar coordenadas para Argentina
    if not (-55 <= lat <= -21.8 and -73.6 <= lon <= -53.6):
        print("⚠️ ADVERTENCIA: Las coordenadas parecen estar fuera de Argentina")
        print("   Rango válido: Lat: -55 a -21.8, Lon: -73.6 a -53.6")
        continuar = input("¿Continuar de todas formas? (s/n): ")
        if continuar.lower() != 's':
            return []
    
    # Obtener datos
    datos = recuperador.obtener_datos_suelo_completo(lat, lon, incluir_datasets)
    
    # Generar reporte
    recuperador.generar_reporte(datos)
    
    # Guardar datos
    if datos:
        recuperador.guardar_datos(datos, lat, lon)
    
    return datos

# Ejemplo de uso
if __name__ == "__main__":
    # Coordenadas de ejemplo - Región agrícola de Córdoba
    LATITUD = -31.4
    LONGITUD = -64.2
    
    # Ejecutar recuperación
    datos_suelo = recuperar_datos_suelo_argentina(LATITUD, LONGITUD)
    
    # Mostrar algunos resultados si se obtuvieron datos
    if datos_suelo:
        print(f"\n🎯 EJEMPLO DE DATOS OBTENIDOS:")
        for i, dato in enumerate(datos_suelo[:3], 1):  # Mostrar primeros 3
            print(f"\n{i}. {dato.get('tipo', 'N/A')} - {dato.get('servicio', 'N/A')}")
            if isinstance(dato, dict):
                for key, value in list(dato.items())[:5]:  # Primeros 5 campos
                    valor_mostrar = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                    print(f"   • {key}: {valor_mostrar}")
    
    print(f"\n🏁 Proceso completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")