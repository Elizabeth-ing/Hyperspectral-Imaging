#CODE 1
# Librerías
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
# Directorio de entrada y nombre base del archivo
input_folder = "C:/Users/Hp/Documents/Internals"
base_name = "internals_mushroom"
# Archivos HDR y RAW
hdr_file = f"{input_folder}/{base_name}.hdr"
raw_file = f"{input_folder}/{base_name}.raw"
# Leer metadatos
metadata = envi.read_envi_header(hdr_file)
# Imprimir metadatos
print("Metadadatos del archivo HDR:")
print(metadata)
# Dimensiones de la imagen
rows = metadata['lines']
cols = metadata['samples']
bands = metadata['bands']
dtype = metadata['data type']
print("\nDimensiones de la imagen:")
print("Filas:", rows)
print("Columnas:", cols)
print("Número de bandas:", bands)
print("Tipo de datos:", dtype)
# Cargar la imagen hiperespectral
img = envi.open(hdr_file, raw_file)
plt.figure()
plt.imshow(np.array(img[:,:,50])) #se visualiza la banda número 50 de 400 bandas
