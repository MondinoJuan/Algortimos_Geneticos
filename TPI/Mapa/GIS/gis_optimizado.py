import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import folium
from folium.plugins import Draw
import webbrowser
import os
import json
import math
import requests
import time
import tempfile

class MapApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Selector de Campo - Argentina")
        self.root.geometry("600x500")
        
        self.coordenadas = []
        self.area_m2 = 0
        self.departamento = ""
        self.provincia = ""
        
        self.create_interface()
        
    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        ttk.Label(main_frame, text="Selector de Campo", 
                 font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Instrucciones compactas
        instructions = ("1. Abrir mapa \n2. Dibujar polígono \n3. Clic derecho \n"
                       "4. Copiar coordenadas (dos corchetes) \n5. Pegar y procesar")
        ttk.Label(main_frame, text=instructions, justify=tk.LEFT, 
                 font=("Arial", 9)).grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)
        
        # Botón mapa
        ttk.Button(main_frame, text="Abrir Mapa", 
                  command=self.open_map).grid(row=2, column=0, columnspan=2, pady=(0, 15))
        
        # Entrada de coordenadas
        ttk.Label(main_frame, text="Coordenadas:").grid(row=3, column=0, sticky=tk.W)
        self.coords_input = scrolledtext.ScrolledText(main_frame, height=4, width=60)
        self.coords_input.grid(row=4, column=0, columnspan=2, pady=(5, 10))
        
        # Botones de acción
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(0, 15))
        
        ttk.Button(btn_frame, text="Procesar", 
                  command=self.process_coordinates).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Limpiar", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Cargar", 
                  command=self.load_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Exportar", 
                  command=self.export_data).pack(side=tk.LEFT)
        
        # Resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.area_var = tk.StringVar(value="Área: No calculada")
        self.location_var = tk.StringVar(value="Ubicación: No determinada")
        
        ttk.Label(results_frame, textvariable=self.area_var, 
                 font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.location_var, 
                 font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W)
        
        # Configurar redimensionamiento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def create_map(self):
        """Crear mapa con controles de dibujo"""
        m = folium.Map(location=[-34.6037, -58.3816], zoom_start=6)
        
        # Plugin de dibujo
        Draw(draw_options={'polygon': True, 'rectangle': True, 'circle': False, 
                          'marker': False, 'circlemarker': False, 'polyline': False},
             edit_options={'edit': True}).add_to(m)
        
        # Script para mostrar coordenadas
        script = """
        <script>
        var drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        
        map.on('draw:created', function(e) {
            var layer = e.layer;
            drawnItems.addLayer(layer);
            
            layer.on('contextmenu', function(e) {
                var coords = layer.getLatLngs()[0];
                var coordsArray = coords.map(c => [c.lat, c.lng]);
                var coordsText = JSON.stringify(coordsArray, null, 2);
                
                L.popup()
                    .setLatLng(e.latlng)
                    .setContent(`<div><strong>Coordenadas:</strong><br>
                               <textarea style="width:280px;height:80px;" readonly>${coordsText}</textarea><br>
                               <small>Ctrl+A para seleccionar todo, Ctrl+C para copiar</small></div>`)
                    .openOn(map);
                    
                setTimeout(() => {
                    const textarea = document.querySelector('textarea');
                    if (textarea) textarea.select();
                }, 100);
            });
        });
        </script>
        """
        m.get_root().html.add_child(folium.Element(script))
        
        return m
        
    def open_map(self):
        """Abrir mapa en navegador usando archivo temporal"""
        try:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                map_path = f.name
                self.create_map().save(map_path)
            
            # Abrir en navegador
            webbrowser.open(f'file://{os.path.abspath(map_path)}')
            
            messagebox.showinfo("Mapa Abierto", 
                "✅ Mapa abierto en navegador\n"
                "• Dibuja polígono\n• Clic derecho sobre él\n• Copia coordenadas")
            
            # Limpiar archivo temporal después de un tiempo
            self.root.after(30000, lambda: self.cleanup_temp_file(map_path))
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el mapa: {str(e)}")
    
    def cleanup_temp_file(self, filepath):
        """Limpiar archivo temporal"""
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except:
            pass  # Ignorar errores de limpieza
            
    def process_coordinates(self):
        """Procesar coordenadas ingresadas"""
        try:
            coords_text = self.coords_input.get("1.0", tk.END).strip()
            if not coords_text:
                messagebox.showwarning("Advertencia", "Ingresa las coordenadas")
                return
            
            # Parsear coordenadas
            coords_text = coords_text.replace('\n', '').replace('\r', '')
            try:
                self.coordenadas = json.loads(coords_text)
            except json.JSONDecodeError:
                coords_text = coords_text.replace('(', '[').replace(')', ']')
                self.coordenadas = json.loads(coords_text)
            
            # Validar
            if not isinstance(self.coordenadas, list) or len(self.coordenadas) < 3:
                raise ValueError("Se necesitan al menos 3 coordenadas")
            
            for coord in self.coordenadas:
                if not isinstance(coord, list) or len(coord) != 2:
                    raise ValueError("Formato: [[-34.123, -58.987], ...]")
                if not (-90 <= coord[0] <= 90) or not (-180 <= coord[1] <= 180):
                    raise ValueError("Coordenadas fuera de rango")
            
            # Calcular área y ubicación
            self.area_m2 = self.calcular_area(self.coordenadas)
            centro_lat = sum(c[0] for c in self.coordenadas) / len(self.coordenadas)
            centro_lon = sum(c[1] for c in self.coordenadas) / len(self.coordenadas)
            self.departamento, self.provincia = self.obtener_ubicacion(centro_lat, centro_lon)
            
            # Actualizar resultados
            hectareas = self.area_m2 / 10000
            if hectareas >= 1:
                area_text = f"Área: {hectareas:.2f} hectáreas ({self.area_m2:.0f} m²)"
            else:
                area_text = f"Área: {self.area_m2:.0f} m²"
            
            self.area_var.set(area_text)
            self.location_var.set(f"Ubicación: {self.departamento}, {self.provincia}")
            
            messagebox.showinfo("Éxito", "¡Área calculada correctamente!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            
    def calcular_area(self, coords):
        """Calcular área usando fórmula de Shoelace"""
        if len(coords) < 3:
            return 0
            
        # Fórmula de Shoelace
        n = len(coords)
        area = sum(coords[i][0] * coords[(i + 1) % n][1] - 
                  coords[(i + 1) % n][0] * coords[i][1] for i in range(n))
        area = abs(area) / 2.0
        
        # Convertir a metros cuadrados
        lat_promedio = sum(c[0] for c in coords) / len(coords)
        factor_lat = math.cos(math.radians(lat_promedio))
        metros_por_grado = 111320
        
        return area * metros_por_grado * metros_por_grado * factor_lat
        
    def obtener_ubicacion(self, lat, lon):
        """Geocodificación inversa"""
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {'lat': lat, 'lon': lon, 'format': 'json', 'addressdetails': 1}
            headers = {'User-Agent': 'CampoSelector/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            address = response.json().get('address', {})
            
            departamento = (address.get('county') or address.get('municipality') or 
                          address.get('city') or 'No determinado')
            provincia = address.get('state') or 'No determinada'
            
            return departamento, provincia
            
        except:
            return "No determinado", "No determinada"
            
    def load_file(self):
        """Cargar coordenadas desde archivo"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON", "*.json"), ("Texto", "*.txt"), ("Todos", "*.*")])
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.coords_input.delete(1.0, tk.END)
                    self.coords_input.insert(1.0, f.read())
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar: {str(e)}")
            
    def export_data(self):
        """Exportar datos a JSON"""
        if not self.coordenadas:
            messagebox.showwarning("Advertencia", "No hay datos para exportar")
            return
            
        try:
            centro_lat = sum(c[0] for c in self.coordenadas) / len(self.coordenadas)
            centro_lon = sum(c[1] for c in self.coordenadas) / len(self.coordenadas)
            
            data = {
                'coordenadas': self.coordenadas,
                'area_m2': self.area_m2,
                'area_hectareas': self.area_m2 / 10000,
                'departamento': self.departamento,
                'provincia': self.provincia,
                'centro': {'lat': centro_lat, 'lon': centro_lon},
                'fecha': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                title="Guardar datos")
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Éxito", f"Exportado a: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar: {str(e)}")
            
    def clear_all(self):
        """Limpiar todo"""
        self.coords_input.delete(1.0, tk.END)
        self.coordenadas = []
        self.area_m2 = 0
        self.area_var.set("Área: No calculada")
        self.location_var.set("Ubicación: No determinada")
        
    def run(self):
        """Ejecutar aplicación"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = MapApp()
        app.run()
    except ImportError as e:
        print(f"Instala dependencias: pip install folium requests")
        print(f"Error: {e}")