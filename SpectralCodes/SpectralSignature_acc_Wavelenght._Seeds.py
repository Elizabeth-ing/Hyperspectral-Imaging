#CODE 2: SPECTRAL SIGNATURE ACCORDING TO WAVELENGTH FOR SEEDS OF THE FRUIT
# Librerías
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np

# Directorio de entrada y nombre base del archivo
input_folder = "C:/Users/Hp/Documents/Internals"    # Ajusta la ruta de los archivos según tu computadora
base_name = "internals_apple"

# Archivos HDR y RAW
hdr_file = f"{input_folder}/{base_name}.hdr"
raw_file = f"{input_folder}/{base_name}.raw"

# Leer metadatos
metadata = envi.read_envi_header(hdr_file)

# Imprimir metadatos
print("Metadatos del archivo HDR:")
print(metadata)

# Dimensiones de la imagen
rows = int(metadata['lines'])
cols = int(metadata['samples'])
bands = int(metadata['bands'])
dtype = metadata['data type']
print("\nDimensiones de la imagen:")
print("Filas:", rows)
print("Columnas:", cols)
print("Número de bandas:", bands)
print("Tipo de datos:", dtype)

# Cargar la imagen hiperespectral
img = envi.open(hdr_file, raw_file).load()

# Verificar las dimensiones de la imagen cargada
print("Dimensiones de la imagen cargada:", img.shape)

# Seleccionar píxeles
pixel_coords = [(633,451), (582,458), (636,478), (579,451), (628,424)] #para las semillas
#pixel_coords = [(748,475), (707,560), (536,567), (508,403), (738,416)] #para la pulpa

# Verificar si las coordenadas están dentro del rango de la imagen
for y, x in pixel_coords:
 if y >= rows or x >= cols:
 print(f"Coordenada {(y, x)} está fuera del rango de la imagen. Dimensiones de la imagen: ({rows}, {cols})")
 continue
 
# Obtener las longitudes de onda desde los metadatos, si están presentes
if 'wavelength' in metadata:
 wavelengths = np.array(metadata['wavelength'], dtype=float)
 print("\nLongitudes de onda (wavelength):")
 print(wavelengths)
else:
 # Usar los índices de las bandas como "longitudes de onda" si no están presentes en los metadatos
 wavelengths = np.arange(bands)
 print("\nLongitudes de onda no disponibles en los metadatos. Usando índices de bandas.")

# Graficar los espectros de los píxeles seleccionados
plt.figure()
for idx, (y, x) in enumerate(pixel_coords):
 spectrum = img[y, x, :].flatten()
 plt.plot(wavelengths, spectrum, label=f'Pixel ({y}, {x})')
plt.xlabel('Longitud de onda' if 'wavelength' in metadata else 'Bandas')
plt.ylabel('Intensidad')
plt.title('Huellas Espectrales de Pixeles Seleccionados Para la Manzana (semillas)')
#plt.title('Huellas Espectrales de Pixeles Seleccionados Para la Manzana (pulpa)')
plt.legend()
plt.show()
