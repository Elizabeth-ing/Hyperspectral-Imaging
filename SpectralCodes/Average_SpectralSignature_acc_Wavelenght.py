# CODE 3
# Librerías
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
# Directorio de entrada y nombre base del archivo
input_folder = "C:/Users/Hp/Documents/Internals"
base_name = "internals_kiwi"
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
# Mostrar una banda específica de la imagen
#plt.figure()
#plt.imshow(np.array(img[:,:,129]))
#plt.title('Banda 129')
#plt.colorbar()
#plt.show()
# Seleccionar píxeles
#pixel_coords = [(535, 363), (736, 454), (659, 530), (687, 318), (708, 388)] #para las semillas
#pixel_coords = [(614, 360), (659, 426), (545, 495), (718, 509), (590, 315)] #para la pulpa
# Verificar si las coordenadas están dentro del rango de la imagen
for y, x in pixel_coords:
 if y >= rows or x >= cols:
 print(f"Coordenada {(y, x)} está fuera del rango de la imagen. Dimensiones de la imagen: ({rows}, {cols})")
 continue
# Extraer los espectros de los píxeles seleccionados
spectra = [img[y, x, :].flatten() for (y, x) in pixel_coords]
# Calcular el promedio de los espectros
average_spectrum = np.mean(spectra, axis=0)
# Obtener las longitudes de onda desde los metadatos, si están presentes
if 'wavelength' in metadata:
 wavelengths = np.array(metadata['wavelength'], dtype=float)
 print("\nLongitudes de onda (wavelength):")
 print(wavelengths)
else:
 # Usar los índices de las bandas como "longitudes de onda" si no están presentes en los metadatos
 wavelengths = np.arange(bands)
 print("\nLongitudes de onda no disponibles en los metadatos. Usando índices de bandas.")
# Graficar el promedio de los espectros de los píxeles seleccionados
plt.figure()
plt.plot(wavelengths, average_spectrum, label='Promedio de Pixeles Seleccionados')
plt.xlabel('Longitud de onda' if 'wavelength' in metadata else 'Bandas')
plt.ylabel('Intensidad')
#plt.title('Huella Espectral Promedio Para la Chirimoya (semillas)')
#plt.title('Huella Espectral Promedio Para la Chirimoya (pulpa)')
plt.legend()
plt.show()
