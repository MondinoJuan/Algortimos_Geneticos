import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

class AnalizadorSueloAgricola:
    """
    Clase mejorada para obtener características específicas del suelo útiles para agricultura
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml, application/json, text/html, */*',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        # Servicios especializados en datos agrícolas y de suelo - ACTUALIZADOS 2025
        self.servicios = {
            'ign_wms': 'https://wms.ign.gob.ar/geoserver/wms',
            'ign_wfs': 'https://wms.ign.gob.ar/geoserver/wfs',
            # INTA con URLs verificadas
            'geointa_wms': 'https://geointa.inta.gob.ar/geoserver/wms',
            'geointa_wfs': 'https://geointa.inta.gob.ar/geoserver/wfs',
            'inta_geo_wms': 'https://geo.inta.gob.ar/geoserver/wms',
            'inta_geo_wfs': 'https://geo.inta.gob.ar/geoserver/wfs',
            # SEGEMAR para datos geológicos relacionados
            'segemar_wms': 'https://sigam.segemar.gov.ar/geoserver/wms',
            'segemar_wfs': 'https://sigam.segemar.gov.ar/geoserver/wfs',
            # SENASA (puede estar inaccesible)
            'senasa_wms': 'https://geonode.senasa.gob.ar/geoserver/wms',
            'senasa_wfs': 'https://geonode.senasa.gob.ar/geoserver/wfs',
        }
        
        # Capas específicas de suelo agrícola conocidas
        self.capas_suelo_prioritarias = {
            'ign': [
                'ign:edafologia_suelos',
                'ign:edafologia_ph',
                'ign:edafologia_textura',
                'ign:edafologia_drenaje',
                'ign:edafologia_capacidad_uso',
                'ign:edafologia_erosion',
                'ign:edafologia_fertilidad',
                'ign:edafologia_profundidad',
                'ign:agricultura_aptitud'
            ],
            'inta': [
                'suelos:capacidad_uso',
                'suelos:ph_suelo',
                'suelos:textura_suelo',
                'suelos:drenaje',
                'suelos:fertilidad',
                'suelos:materia_organica',
                'agricultura:aptitud_agricola',
                'agricultura:limitaciones',
                'clima:precipitaciones',
                'clima:temperatura'
            ],
            'senasa': [
                'agricultura:zonas_productivas',
                'sanidad:limitaciones_fitosanitarias',
                'cultivos:aptitud_por_cultivo'
            ]
        }
    
    def obtener_capas_suelo_especificas(self, servicio_wms):
        """Obtiene capas específicamente relacionadas con agricultura y suelo"""
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
                    capas_agricultura = []
                    
                    for layer in root.iter():
                        if layer.tag.endswith('Layer') or layer.tag == 'Layer':
                            name_elem = None
                            title_elem = None
                            abstract_elem = None
                            
                            for child in layer:
                                if child.tag.endswith('Name') or child.tag == 'Name':
                                    name_elem = child
                                elif child.tag.endswith('Title') or child.tag == 'Title':
                                    title_elem = child
                                elif child.tag.endswith('Abstract') or child.tag == 'Abstract':
                                    abstract_elem = child
                            
                            if name_elem is not None and name_elem.text:
                                name = name_elem.text.strip()
                                title = title_elem.text.strip() if title_elem is not None and title_elem.text else name
                                abstract = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else ""
                                
                                # Palabras clave específicas para análisis agrícola
                                keywords_agricultura = [
                                    # Propiedades físicas del suelo
                                    'ph', 'textura', 'texture', 'clay', 'sand', 'silt', 'arcilla', 'arena', 'limo',
                                    'drenaje', 'drainage', 'permeabilidad', 'infiltracion',
                                    'densidad', 'porosidad', 'compactacion',
                                    
                                    # Propiedades químicas
                                    'fertilidad', 'fertility', 'nutrient', 'nutriente', 'nitrogen', 'nitrogeno',
                                    'phosphor', 'fosforo', 'potassium', 'potasio', 'calcium', 'calcio',
                                    'materia_organica', 'organic_matter', 'carbon', 'carbono',
                                    'salinidad', 'salinity', 'sodio', 'conductividad',
                                    
                                    # Aptitud agrícola
                                    'aptitud', 'suitability', 'capacidad', 'capacity', 'uso', 'use',
                                    'agricultura', 'agricultural', 'cultivo', 'crop', 'siembra', 'farming',
                                    'productividad', 'productivity', 'rendimiento', 'yield',
                                    
                                    # Limitaciones
                                    'erosion', 'limitation', 'limitacion', 'riesgo', 'risk',
                                    'pendiente', 'slope', 'inundacion', 'flood',
                                    
                                    # Clasificación de suelos
                                    'serie', 'series', 'tipo', 'type', 'class', 'orden', 'taxonomy',
                                    'mollisol', 'entisol', 'vertisol', 'alfisol'
                                ]
                                
                                # Filtrar capas no relacionadas con demografía/estadísticas
                                keywords_excluir = [
                                    'eph', 'encuesta', 'poblacion', 'censo', 'habitantes', 'vivienda',
                                    'aglomerado', 'localidad', 'entidades', 'administrative', 'boundary'
                                ]
                                
                                # Verificar si la capa es relevante para agricultura
                                texto_completo = f"{name.lower()} {title.lower()} {abstract.lower()}"
                                
                                es_agricultura = any(keyword in texto_completo for keyword in keywords_agricultura)
                                es_excluir = any(keyword in texto_completo for keyword in keywords_excluir)
                                
                                if es_agricultura and not es_excluir:
                                    capas_agricultura.append({
                                        'nombre': name,
                                        'titulo': title,
                                        'resumen': abstract,
                                        'relevancia': self._calcular_relevancia(texto_completo, keywords_agricultura)
                                    })
                    
                    # Ordenar por relevancia
                    capas_agricultura.sort(key=lambda x: x['relevancia'], reverse=True)
                    return capas_agricultura
                    
                except ET.ParseError:
                    return []
            return []
        except Exception as e:
            print(f"Error obteniendo capas de {servicio_wms}: {e}")
            return []
    
    def _calcular_relevancia(self, texto, keywords):
        """Calcula un puntaje de relevancia basado en palabras clave"""
        puntaje = 0
        for keyword in keywords:
            if keyword in texto:
                # Dar más puntos a términos más específicos
                if keyword in ['ph', 'textura', 'fertilidad', 'drenaje', 'aptitud']:
                    puntaje += 3
                elif keyword in ['agricultura', 'cultivo', 'suelo']:
                    puntaje += 2
                else:
                    puntaje += 1
        return puntaje
    
    def consultar_caracteristicas_punto(self, servicio_wms, capa, lat, lon):
        """Consulta características específicas del suelo en un punto"""
        try:
            params = {
                'SERVICE': 'WMS',
                'VERSION': '1.1.1',
                'REQUEST': 'GetFeatureInfo',
                'LAYERS': capa,
                'QUERY_LAYERS': capa,
                'SRS': 'EPSG:4326',
                'BBOX': f"{lon-0.001},{lat-0.001},{lon+0.001},{lat+0.001}",
                'WIDTH': 3,
                'HEIGHT': 3,
                'X': 1,
                'Y': 1,
                'INFO_FORMAT': 'application/json',  # Preferir JSON
                'FEATURE_COUNT': 5
            }
            
            response = requests.get(servicio_wms, params=params, headers=self.headers, timeout=20, verify=False)
            
            if response.status_code == 200:
                try:
                    # Intentar parsear como JSON primero
                    data = response.json()
                    if data.get('features'):
                        return {'tipo': 'json', 'datos': data['features']}
                except:
                    # Si falla JSON, usar texto plano
                    if response.text.strip() and "no features were found" not in response.text.lower():
                        return {'tipo': 'texto', 'datos': response.text.strip()}
            
            return None
            
        except Exception as e:
            return None
    
    def extraer_propiedades_suelo(self, datos):
        """Extrae y clasifica propiedades específicas del suelo"""
        propiedades_suelo = {
            'fisicas': {},
            'quimicas': {},
            'biologicas': {},
            'aptitud_agricola': {},
            'limitaciones': {}
        }
        
        if datos['tipo'] == 'json':
            for feature in datos['datos']:
                if 'properties' in feature:
                    props = feature['properties']
                    self._clasificar_propiedades(props, propiedades_suelo)
        
        elif datos['tipo'] == 'texto':
            # Parsear texto plano buscando propiedades conocidas
            texto = datos['datos'].lower()
            if 'ph' in texto:
                propiedades_suelo['quimicas']['informacion_ph'] = datos['datos']
            if any(term in texto for term in ['textura', 'clay', 'sand', 'arcilla', 'arena']):
                propiedades_suelo['fisicas']['informacion_textura'] = datos['datos']
            if any(term in texto for term in ['drenaje', 'drainage']):
                propiedades_suelo['fisicas']['informacion_drenaje'] = datos['datos']
        
        return propiedades_suelo
    
    def _clasificar_propiedades(self, props, propiedades_suelo):
        """Clasifica propiedades en categorías relevantes para agricultura"""
        for key, value in props.items():
            if not value or str(value).strip() == 'null':
                continue
                
            key_lower = key.lower()
            
            # Propiedades físicas
            if any(term in key_lower for term in ['textura', 'texture', 'clay', 'sand', 'silt', 'arcilla', 'arena', 'limo']):
                propiedades_suelo['fisicas'][key] = value
            elif any(term in key_lower for term in ['drenaje', 'drainage', 'permeabilidad', 'infiltracion']):
                propiedades_suelo['fisicas'][key] = value
            elif any(term in key_lower for term in ['densidad', 'density', 'porosidad', 'compactacion']):
                propiedades_suelo['fisicas'][key] = value
            
            # Propiedades químicas
            elif any(term in key_lower for term in ['ph', 'acidez', 'alcalinidad']):
                propiedades_suelo['quimicas'][key] = value
            elif any(term in key_lower for term in ['fertilidad', 'fertility', 'nutrient', 'nutriente']):
                propiedades_suelo['quimicas'][key] = value
            elif any(term in key_lower for term in ['nitrogen', 'nitrogeno', 'phosphor', 'fosforo', 'potassium', 'potasio']):
                propiedades_suelo['quimicas'][key] = value
            elif any(term in key_lower for term in ['materia_organica', 'organic', 'carbon', 'carbono']):
                propiedades_suelo['biologicas'][key] = value
            elif any(term in key_lower for term in ['salinidad', 'salinity', 'conductividad']):
                propiedades_suelo['quimicas'][key] = value
            
            # Aptitud agrícola
            elif any(term in key_lower for term in ['aptitud', 'suitability', 'capacidad', 'uso', 'agricultura']):
                propiedades_suelo['aptitud_agricola'][key] = value
            elif any(term in key_lower for term in ['cultivo', 'crop', 'productividad', 'rendimiento']):
                propiedades_suelo['aptitud_agricola'][key] = value
            
            # Limitaciones
            elif any(term in key_lower for term in ['erosion', 'limitation', 'limitacion', 'riesgo', 'pendiente']):
                propiedades_suelo['limitaciones'][key] = value
            
            # Si no encaja en ninguna categoría específica pero parece relevante
            elif any(term in key_lower for term in ['suelo', 'soil', 'tierra', 'land']):
                propiedades_suelo['fisicas'][key] = value
    
    def analizar_suelo_para_siembra(self, lat, lon):
        """
        Función principal para análisis de suelo específico para siembra
        """
        print(f"🌱 ANÁLISIS DE SUELO PARA SIEMBRA")
        print(f"📍 Coordenadas: ({lat}, {lon})")
        print(f"🕐 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        analisis = {
            'coordenadas': {'latitud': lat, 'longitud': lon},
            'fecha_analisis': datetime.now().isoformat(),
            'propiedades_encontradas': {
                'fisicas': {},
                'quimicas': {},
                'biologicas': {},
                'aptitud_agricola': {},
                'limitaciones': {}
            },
            'recomendaciones': [],
            'fuentes_consultadas': []
        }
        
        # 1. Consultar IGN con capas específicas de suelo
        print("\n🗺️ Analizando datos del IGN...")
        capas_ign = self.obtener_capas_suelo_especificas(self.servicios['ign_wms'])
        
        if capas_ign:
            print(f"✅ Encontradas {len(capas_ign)} capas de suelo relevantes")
            
            for capa in capas_ign[:8]:  # Analizar las 8 más relevantes
                nombre_capa = capa['nombre']
                titulo_capa = capa['titulo']
                
                print(f"🔍 Analizando: {titulo_capa} (relevancia: {capa['relevancia']})")
                
                datos = self.consultar_caracteristicas_punto(self.servicios['ign_wms'], nombre_capa, lat, lon)
                
                if datos:
                    propiedades = self.extraer_propiedades_suelo(datos)
                    
                    # Integrar propiedades encontradas
                    for categoria, props in propiedades.items():
                        if props:
                            analisis['propiedades_encontradas'][categoria].update(props)
                            print(f"  ✅ Propiedades {categoria}: {len(props)} encontradas")
                    
                    analisis['fuentes_consultadas'].append(f"IGN - {titulo_capa}")
        
        # 2. Consultar INTA (más específico para agricultura)
        print(f"\n🌾 Analizando datos del INTA (múltiples fuentes)...")
        
        # Probar con diferentes endpoints de INTA
        servicios_inta = ['geointa_wms', 'inta_geo_wms']
        
        for servicio_inta in servicios_inta:
            if servicio_inta in self.servicios:
                print(f"🔍 Probando {servicio_inta}...")
                
                try:
                    capas_inta = self.obtener_capas_suelo_especificas(self.servicios[servicio_inta])
                    
                    if capas_inta:
                        print(f"✅ Encontradas {len(capas_inta)} capas agrícolas en {servicio_inta}")
                        
                        for capa in capas_inta[:5]:
                            nombre_capa = capa['nombre']
                            titulo_capa = capa['titulo']
                            
                            print(f"🔍 Analizando INTA: {titulo_capa}")
                            
                            datos = self.consultar_caracteristicas_punto(self.servicios[servicio_inta], nombre_capa, lat, lon)
                            
                            if datos:
                                propiedades = self.extraer_propiedades_suelo(datos)
                                
                                for categoria, props in propiedades.items():
                                    if props:
                                        analisis['propiedades_encontradas'][categoria].update(props)
                                        print(f"  ✅ INTA - Propiedades {categoria}: {len(props)}")
                                
                                analisis['fuentes_consultadas'].append(f"INTA - {titulo_capa}")
                        break  # Si encontramos datos, no probar otros servicios INTA
                    else:
                        print(f"⚠️ {servicio_inta} sin capas relevantes")
                        
                except Exception as e:
                    print(f"❌ Error conectando con {servicio_inta}: {e}")
        
        # 3. Consultar SEGEMAR para datos geológicos relacionados
        print(f"\n🗿 Analizando datos geológicos (SEGEMAR)...")
        try:
            capas_segemar = self.obtener_capas_suelo_especificas(self.servicios['segemar_wms'])
            
            if capas_segemar:
                print(f"✅ Encontradas {len(capas_segemar)} capas geológicas relevantes")
                
                for capa in capas_segemar[:3]:
                    nombre_capa = capa['nombre']
                    titulo_capa = capa['titulo']
                    
                    datos = self.consultar_caracteristicas_punto(self.servicios['segemar_wms'], nombre_capa, lat, lon)
                    
                    if datos:
                        propiedades = self.extraer_propiedades_suelo(datos)
                        
                        for categoria, props in propiedades.items():
                            if props:
                                analisis['propiedades_encontradas'][categoria].update(props)
                        
                        analisis['fuentes_consultadas'].append(f"SEGEMAR - {titulo_capa}")
            else:
                print("⚠️ SEGEMAR sin capas relevantes para suelos")
        except Exception as e:
            print(f"❌ SEGEMAR no accesible: {e}")
        
        # 4. Consultar SENASA para limitaciones fitosanitarias
        print(f"\n🛡️ Analizando limitaciones fitosanitarias (SENASA)...")
        try:
            capas_senasa = self.obtener_capas_suelo_especificas(self.servicios['senasa_wms'])
            
            if capas_senasa:
                print(f"✅ Encontradas {len(capas_senasa)} capas en SENASA")
                
                for capa in capas_senasa[:3]:
                    nombre_capa = capa['nombre']
                    titulo_capa = capa['titulo']
                    
                    datos = self.consultar_caracteristicas_punto(self.servicios['senasa_wms'], nombre_capa, lat, lon)
                    
                    if datos:
                        propiedades = self.extraer_propiedades_suelo(datos)
                        
                        for categoria, props in propiedades.items():
                            if props:
                                analisis['propiedades_encontradas'][categoria].update(props)
                        
                        analisis['fuentes_consultadas'].append(f"SENASA - {titulo_capa}")
            else:
                print("⚠️ SENASA temporalmente no disponible")
        except Exception as e:
            print(f"❌ SENASA no accesible: {e}")
        
        # Generar recomendaciones básicas
        analisis['recomendaciones'] = self._generar_recomendaciones(analisis['propiedades_encontradas'])
        
        return analisis
    
    def _generar_recomendaciones(self, propiedades):
        """Genera recomendaciones básicas basadas en las propiedades encontradas"""
        recomendaciones = []
        
        # Verificar si hay suficientes datos para recomendaciones
        total_propiedades = sum(len(props) for props in propiedades.values())
        
        if total_propiedades == 0:
            recomendaciones.append("No se encontraron datos específicos de suelo. Se recomienda:")
            recomendaciones.append("• Realizar análisis de suelo in situ")
            recomendaciones.append("• Consultar con ingeniero agrónomo local")
            recomendaciones.append("• Verificar con productores de la zona")
        else:
            recomendaciones.append("Basado en los datos encontrados:")
            
            if propiedades['quimicas']:
                recomendaciones.append("• Se encontraron datos químicos del suelo - analizar pH y fertilidad")
            
            if propiedades['fisicas']:
                recomendaciones.append("• Se encontraron datos físicos - verificar textura y drenaje")
            
            if propiedades['aptitud_agricola']:
                recomendaciones.append("• Hay información de aptitud agrícola disponible")
            
            if propiedades['limitaciones']:
                recomendaciones.append("• IMPORTANTE: Revisar limitaciones identificadas")
            
            recomendaciones.append("• Complementar con análisis de laboratorio")
            recomendaciones.append("• Considerar condiciones climáticas locales")
        
        return recomendaciones
    
    def generar_reporte_agricola(self, analisis):
        """Genera reporte específico para análisis agrícola"""
        
        print(f"\n🌱 REPORTE DE ANÁLISIS DE SUELO PARA SIEMBRA")
        print("="*70)
        print(f"📍 Ubicación: {analisis['coordenadas']['latitud']}, {analisis['coordenadas']['longitud']}")
        print(f"🕐 Fecha análisis: {analisis['fecha_analisis'][:19]}")
        
        propiedades = analisis['propiedades_encontradas']
        total_datos = sum(len(props) for props in propiedades.values())
        
        print(f"📊 Total de propiedades encontradas: {total_datos}")
        print(f"🗂️ Fuentes consultadas: {len(analisis['fuentes_consultadas'])}")
        
        if total_datos == 0:
            print(f"\n❌ NO SE ENCONTRARON DATOS ESPECÍFICOS DE SUELO")
            print(f"\n💡 POSIBLES CAUSAS:")
            print(f"• La zona no tiene estudios de suelo digitalizados")
            print(f"• Los servicios no tienen cobertura en esta área")
            print(f"• Las coordenadas no corresponden a zonas agrícolas")
            
            print(f"\n🔧 ACCIONES RECOMENDADAS:")
            for rec in analisis['recomendaciones']:
                print(f"  {rec}")
            
            return analisis
        
        # Mostrar propiedades por categoría
        categorias = {
            'fisicas': '🏗️ PROPIEDADES FÍSICAS',
            'quimicas': '⚗️ PROPIEDADES QUÍMICAS', 
            'biologicas': '🦠 PROPIEDADES BIOLÓGICAS',
            'aptitud_agricola': '🌾 APTITUD AGRÍCOLA',
            'limitaciones': '⚠️ LIMITACIONES'
        }
        
        for categoria, titulo in categorias.items():
            props = propiedades[categoria]
            if props:
                print(f"\n{titulo}")
                print("-" * len(titulo))
                
                for key, value in props.items():
                    # Formatear valor para mostrar
                    if isinstance(value, str) and len(value) > 100:
                        valor_mostrar = value[:100] + "..."
                    else:
                        valor_mostrar = str(value)
                    
                    print(f"• {key}: {valor_mostrar}")
        
        # Mostrar recomendaciones
        if analisis['recomendaciones']:
            print(f"\n🎯 RECOMENDACIONES PARA SIEMBRA")
            print("-" * 35)
            for rec in analisis['recomendaciones']:
                print(f"{rec}")
        
        # Mostrar fuentes
        if analisis['fuentes_consultadas']:
            print(f"\n📚 FUENTES DE DATOS CONSULTADAS")
            print("-" * 32)
            for i, fuente in enumerate(analisis['fuentes_consultadas'], 1):
                print(f"{i}. {fuente}")
        
        return analisis
    
    def guardar_analisis(self, analisis, lat, lon):
        """Guarda el análisis en archivos optimizados para agricultura"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"analisis_suelo_agricola_{lat}_{lon}_{timestamp}"
        
        # Guardar JSON completo
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analisis, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Análisis completo guardado en: {json_file}")
        
        # Crear resumen CSV especializado para agricultura
        try:
            datos_csv = []
            
            # Resumen por categorías
            propiedades = analisis['propiedades_encontradas']
            
            fila_resumen = {
                'coordenadas': f"{lat}, {lon}",
                'fecha_analisis': analisis['fecha_analisis'][:10],
                'propiedades_fisicas': len(propiedades['fisicas']),
                'propiedades_quimicas': len(propiedades['quimicas']),
                'propiedades_biologicas': len(propiedades['biologicas']),
                'aptitud_agricola': len(propiedades['aptitud_agricola']),
                'limitaciones': len(propiedades['limitaciones']),
                'total_propiedades': sum(len(props) for props in propiedades.values()),
                'fuentes_consultadas': len(analisis['fuentes_consultadas']),
                'apto_para_siembra': 'SI' if sum(len(props) for props in propiedades.values()) > 0 else 'REQUIERE_ANALISIS'
            }
            
            datos_csv.append(fila_resumen)
            
            df = pd.DataFrame(datos_csv)
            csv_file = f"{base_filename}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"📊 Resumen agrícola guardado en: {csv_file}")
            
            return json_file, csv_file
            
        except Exception as e:
            print(f"⚠️ Error creando CSV: {e}")
            return json_file, None

def analizar_suelo_para_siembra(latitud, longitud):
    """
    Función principal para análisis de suelo específico para siembra
    
    Args:
        latitud (float): Latitud en grados decimales
        longitud (float): Longitud en grados decimales
    
    Returns:
        dict: Diccionario con análisis específico para agricultura
    """
    
    # Validar coordenadas
    if not (-90 <= latitud <= 90) or not (-180 <= longitud <= 180):
        print("❌ Error: Coordenadas inválidas")
        return None
    
    # Verificar si está en Argentina (opcional)
    if not (-55 <= latitud <= -21.8 and -73.6 <= longitud <= -53.6):
        print("⚠️ ADVERTENCIA: Coordenadas fuera de Argentina")
        print("   Sistema optimizado para suelos argentinos")
    
    # Crear analizador especializado
    analizador = AnalizadorSueloAgricola()
    
    # Realizar análisis
    analisis = analizador.analizar_suelo_para_siembra(latitud, longitud)
    
    # Generar reporte
    analizador.generar_reporte_agricola(analisis)
    
    # Guardar archivos
    analizador.guardar_analisis(analisis, latitud, longitud)
    
    return analisis

# Ejemplo de uso optimizado para agricultura
if __name__ == "__main__":
    print("🌱 SISTEMA DE ANÁLISIS DE SUELO PARA SIEMBRA")
    print("🎯 Versión especializada en datos agrícolas")
    print("="*70)
    
    # Coordenadas de zonas agrícolas importantes de Argentina
    coordenadas_agricolas = [
        (-33.7577, -61.9567, "Zona Núcleo - Pergamino, Buenos Aires"),
        (-32.9442, -60.6505, "Rosario - Santa Fe (zona sojera)"),
        (-31.7333, -60.5333, "Entre Ríos - zona agrícola"),
        (-34.0144, -59.8628, "San Nicolás - Buenos Aires"),
        (-31.4201, -64.1888, "Córdoba - zona mixta"),
    ]
    
    print(f"🔍 PROBANDO COORDENADAS EN ZONAS AGRÍCOLAS CONOCIDAS")
    
    for lat, lon, descripcion in coordenadas_agricolas:
        print(f"\n{'='*50}")
        print(f"📍 ANALIZANDO: {descripcion}")
        print(f"🗺️ Coordenadas: {lat}, {lon}")
        
        resultado = analizar_suelo_para_siembra(lat, lon)
        
        if resultado:
            total_props = sum(len(props) for props in resultado['propiedades_encontradas'].values())
            if total_props > 0:
                print(f"✅ ¡DATOS AGRÍCOLAS ENCONTRADOS! ({total_props} propiedades)")
                break
            else:
                print(f"⚠️ Sin datos específicos en {descripcion}")
        
        print(f"Esperando antes de la siguiente consulta...")
        import time
        time.sleep(2)  # Evitar sobrecargar los servicios
    
    # Si no se encontraron datos en las coordenadas de prueba
    print(f"\n🎯 EJEMPLO COMPLETO CON COORDENADAS ESPECÍFICAS")
    LATITUD = -33.7577  # Pergamino, Buenos Aires - zona núcleo agrícola
    LONGITUD = -61.9567
    print(f"📍 Analizando zona agrícola: Pergamino, Buenos Aires")
    print(f"🗺️ Coordenadas: {LATITUD}, {LONGITUD}")
    
    resultado_final = analizar_suelo_para_siembra(LATITUD, LONGITUD)
    
    if resultado_final:
        total_propiedades = sum(len(props) for props in resultado_final['propiedades_encontradas'].values())
        
        if total_propiedades > 0:
            print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print(f"📊 Se encontraron {total_propiedades} propiedades específicas de suelo")
            print(f"🌾 Datos útiles para análisis de siembra")
            
            # Mostrar resumen de lo encontrado
            props = resultado_final['propiedades_encontradas']
            if props['quimicas']:
                print(f"⚗️ Propiedades químicas: {len(props['quimicas'])} (pH, fertilidad, nutrientes)")
            if props['fisicas']:
                print(f"🏗️ Propiedades físicas: {len(props['fisicas'])} (textura, drenaje)")
            if props['aptitud_agricola']:
                print(f"🌱 Aptitud agrícola: {len(props['aptitud_agricola'])} (capacidad de uso)")
            if props['limitaciones']:
                print(f"⚠️ Limitaciones: {len(props['limitaciones'])} (erosión, pendientes)")
                
        else:
            print(f"\n⚠️ NO SE ENCONTRARON DATOS ESPECÍFICOS DE SUELO")
            print(f"💡 RECOMENDACIONES ALTERNATIVAS:")
            print(f"• Contactar INTA regional para datos locales")
            print(f"• Realizar análisis de suelo en laboratorio")
            print(f"• Consultar con productores locales")
            print(f"• Verificar cartas de suelo del INTA en formato papel")
    
    print(f"\n🏁 PROCESO FINALIZADO")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📚 RECURSOS ADICIONALES RECOMENDADOS:")
    print(f"• INTA: https://inta.gob.ar/suelos")
    print(f"• Cartas de Suelo: https://inta.gob.ar/documentos/cartas-de-suelos")
    print(f"• Sistema de Información de Suelos: https://sisinta.inta.gob.ar/")