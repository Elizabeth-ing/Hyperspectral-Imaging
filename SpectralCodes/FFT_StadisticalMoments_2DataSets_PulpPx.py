# CODE 6: FOURIER TRANSFORM AND STATISTICAL MOMENTS FOR 2 DATA SETS OF PULP PIXEL
# Librerías
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
# Directorio de entrada y nombre base del archivo
input_folder = "C:/Users/Hp/Documents/Internals"
base_name = "internals_papaya"
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
#PAPAYA
"""pixel_coords = [
 (409, 456), (421, 465), (429, 468), (425, 460), (408, 451),
 (426, 459), (426, 444), (430, 452), (434, 460), (437, 468),
 (443, 477), (440, 466), (442, 454), (438, 443), (435, 435),
 (440, 444), (442, 452), (442, 460), (442, 468), (441, 471),
 (443, 479), (447, 470), (447, 478), (446, 443), (445, 433),
 (447, 422), (452, 429), (449, 418), (451, 409), (455, 433),
 (459, 447), (466, 455), (465, 414), (465, 397), (468, 403),
 (478, 400), (476, 386), (481, 396), (481, 404), (482, 413),
 (482, 421), (482, 412), (482, 393), (497, 383), (500, 397),
 (506, 394), (515, 394), (529, 400), (515, 427), (508, 440)
]"""
#PAPAYA 
pixel_coords = [
 (510,451), (514,458), (525,450), (525,456), (536,464),
 (538,488), (537,503), (547,495), (537,472), (542,446),
 (548,422), (542,405), (551,389), (555,401), (555,380),
 (565,349), (507,403), (569,415), (583,415), (584,386),
 (583,397), (583,388), (578,380), (587,390), (587,399),
 (606,402), (611,389), (616,403), (625,408), (628,418),
 (633,428), (633,438), (631,452), (620,462), (615,475),
 (618,484), (614,513), (513,634), (624,413), (631,398),
 (634,380), (641,391), (642,402), (642,417), (637,428),
 (638,440), (638,452), (635,459), (635,465), (634,476)
]
# Verificar si las coordenadas están dentro del rango de la imagen
valid_coords = []
for coord in pixel_coords:
 y, x = coord
 if y < rows and x < cols:
 valid_coords.append((y, x))
 else:
 print(f"Coordenada {(y, x)} está fuera del rango de la imagen. Dimensiones de la imagen: ({rows}, {cols})")
# Extraer los espectros de los píxeles seleccionados
spectra = [img[y, x, :].flatten() for (y, x) in valid_coords]
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
plt.title('Huella Espectral Promedio (semillas)')
plt.legend()
plt.show()
# Calcular la transformada de Fourier de la huella espectral promedio
fft_spectrum = np.fft.fft(average_spectrum)
fft_freq = np.fft.fftfreq(len(average_spectrum), d=(wavelengths[1] - wavelengths[0]) if len(wavelengths) > 1 else 1)
plt.figure()
plt.plot(fft_freq, np.abs(fft_spectrum), label='Transformada de Fourier')
plt.xlabel('Frecuencia')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier de la Huella Espectral Promedio (semillas)')
plt.legend()
plt.show()
# Calcular los momentos estadísticos de la huella espectral promedio
mean_value = np.mean(average_spectrum)
variance_value = np.var(average_spectrum)
skewness_value = skew(average_spectrum)
kurtosis_value = kurtosis(average_spectrum)
# Imprimir los momentos estadísticos
print("\nMomentos estadísticos de la huella espectral promedio:")
print(f"Media: {mean_value}")
print(f"Varianza: {variance_value}")
print(f"Asimetría: {skewness_value}")
print(f"Curtosis: {kurtosis_value}")
