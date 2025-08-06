import pandas as pd

# Cargar los CSV
sites = pd.read_csv('Bases de datos/Suelo/SISLAC_site.csv')
layers = pd.read_csv('Bases de datos/Suelo/SISLAC_layers.csv')

# Elegir solo algunas columnas
sites_collums = sites[['profile_identifier', 'latitude', 'longitude', 'country_code']]
layers_collums = layers[['profile_identifier', 'bulk_density', 'ca_co3', 'coarse_fragments', 'ecec', 'conductivity', 
                          'organic_carbon', 'ph', 'clay', 'silt', 'sand', 'water_retention']]

# Filtro datos innesesarios
sites_collums = sites_collums[sites_collums['country_code'] == 'ARG']

# Hacer un join (merge) usando site_id
merged = pd.merge(sites_collums, layers_collums, on='profile_identifier', how='inner')

# Mostrar resultado
print(merged.head())

# Guardar si querés
merged.to_csv('Recuperacion de datos/Suelos/suelo_unido.csv', index=False)