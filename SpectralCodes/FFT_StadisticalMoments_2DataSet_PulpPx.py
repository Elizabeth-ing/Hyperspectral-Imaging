# CODE 5: FOURIER TRANSFORM AND STATISTICAL MOMENTS FOR 2 DATA SETS OF PULP PIXEL
# Librerías
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# Directorio de entrada y nombre base del archivo
input_folder = "C:/Users/Hp/Documents/Internals"    # Ajusta la ruta de los archivos según tu computadora
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
plt.figure()
plt.imshow(np.array(img[:,:,129]))
plt.title('Banda 129')
plt.colorbar()
plt.show()

#PAPAYA
pixel_coords = [
 (258,446), (290,480), (256,416), (268,376), (297,370),
 (305,421), (315,375), (337,345), (359,390), (351,431),
 (351,463), (346,500), (434,522), (371,502), (408,488),
 (413,527), (432,495), (420,519), (386,399), (386,336),
 (425,358), (440,377), (316,440), (459,350), (481,318),
 (511,333), (552,345), (557,311), (570,331), (604,316),
 (631,336), (660,321), (377,291), (690,328), (709,304),
 (334,313), (734,336), (756,306), (766,343), (790,328),
 (805,348), (825,348), (825,377), (844,365), (857,370),
 (857,394), (812,399), (805,421), (842,416), (876,416)
]

#PAPAYA
"""pixel_coords = [
 (893,416), (859,429), (815,431), (827,448), (859,448),
 (891,463), (903,483), (403,401), (879,515), (852,542),
 (834,534), (800,527), (800,542), (758,515), (788,583),
 (751,593), (731,605), (707,583), (707,576), (704,598),
 (670,537), (646,537), (638,531), (864,382), (864,399),
 (430,342), (373,333), (373,345), (795,392), (800,424),
 (822,424), (812,333), (528,517), (422,483), (418,475),
 (445,392), (220,428), (228,394), (378,240), (251,368),
 (236,483), (255,359), (407,320), (440,303), (449,318),
 (504,293), (494,316), (518,342), (557,292), (552,329)
]"""

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
plt.title('Huella Espectral Promedio (pulpa)')
plt.legend()
plt.show()

# Calcular la transformada de Fourier de la huella espectral promedio
fft_spectrum = np.fft.fft(average_spectrum)
fft_freq = np.fft.fftfreq(len(average_spectrum), d=(wavelengths[1] - wavelengths[0]) if len(wavelengths) > 1 else 1)
plt.figure()
plt.plot(fft_freq, np.abs(fft_spectrum), label='Transformada de Fourier')
plt.xlabel('Frecuencia')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier de la Huella Espectral Promedio (pulpa)')
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
