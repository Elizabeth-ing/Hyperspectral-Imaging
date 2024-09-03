# CODE 8: KNN AND DECISION BOUNDARY
# Librerías
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# Coordenadas de las semillas y la pulpa
seed_coords = [
 (409, 456), (421, 465), (429, 468), (425, 460), (408, 451),
 (426, 459), (426, 444), (430, 452), (434, 460), (437, 468),
 (443, 477), (440, 466), (442, 454), (438, 443), (435, 435),
 (440, 444), (442, 452), (442, 460), (442, 468), (441, 471),
 (443, 479), (447, 470), (447, 478), (446, 443), (445, 433),
 (447, 422), (452, 429), (449, 418), (451, 409), (455, 433),
 (459, 447), (466, 455), (465, 414), (465, 397), (468, 403),
 (478, 400), (476, 386), (481, 396), (481, 404), (482, 413),
 (482, 421), (482, 412), (482, 393), (497, 383), (500, 397),
 (506, 394), (515, 394), (529, 400), (515, 427), (508, 440),
 (510, 451), (514, 458), (525, 450), (525, 456), (536, 464),
 (538, 488), (537, 503), (547, 495), (537, 472), (542, 446),
 (548, 422), (542, 405), (551, 389), (555, 401), (555, 380),
 (565, 349), (507, 403), (569, 415), (583, 415), (584, 386),
 (583, 397), (583, 388), (578, 380), (587, 390), (587, 399),
 (606, 402), (611, 389), (616, 403), (625, 408), (628, 418),
 (633, 428), (633, 438), (631, 452), (620, 462), (615, 475),
 (618, 484), (614, 513), (513, 634), (624, 413), (631, 398),
 (634, 380), (641, 391), (642, 402), (642, 417), (637, 428),
 (638, 440), (638, 452), (635, 459), (635, 465), (634, 476)
]
pulp_coords = [
 (258,446), (290,480), (256,416), (268,376), (297,370),
 (305,421), (315,375), (337,345), (359,390), (351,431),
 (351,463), (346,500), (434,522), (371,502), (408,488),
 (413,527), (432,495), (420,519), (386,399), (386,336),
 (425,358), (440,377), (316,440), (459,350), (481,318),
 (511,333), (552,345), (557,311), (570,331), (604,316),
 (631,336), (660,321), (377,291), (690,328), (709,304),
 (334,313), (734,336), (756,306), (766,343), (790,328),
 (805,348), (825,348), (825,377), (844,365), (857,370),
 (857,394), (812,399), (805,421), (842,416), (876,416),
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
]

# Verificar si las coordenadas están dentro del rango de la imagen
valid_seed_coords = [(y, x) for y, x in seed_coords if y < rows and x < cols]
valid_pulp_coords = [(y, x) for y, x in pulp_coords if y < rows and x < cols]

# Etiquetas: 0 = semillas, 1 = pulpa
labels_seed = [0] * len(valid_seed_coords)
labels_pulp = [1] * len(valid_pulp_coords)

# Unión de los datos y etiquetas
coords = valid_seed_coords + valid_pulp_coords
labels = labels_seed + labels_pulp

# Extraer los espectros de los píxeles seleccionados
spectra = [img[y, x, :].flatten() for (y, x) in coords]

# Calcular los momentos estadísticos para cada espectro
def compute_moments(spectrum):
 return [
 np.mean(spectrum),
 np.std(spectrum),
 skew(spectrum),
 kurtosis(spectrum)
 ]

# Aplicar la función a todos los espectros
moments = [compute_moments(spectrum) for spectrum in spectra]
# Datos de entrenamiento y prueba (75 entrenamiento, 25 prueba por clase)
X = np.array(moments)
y = np.array(labels)

# Seleccionar 75 datos de entrenamiento y 25 de prueba por clase
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Crear una malla de puntos para graficar las fronteras de decisión
h = 1 # paso de la malla

# Graficar los pares de momentos estadísticos
plt.figure(figsize=(15, 12))

# Media vs Desviación Estándar
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 2])), np.full_like(xx.ravel(), np.mean(X[:,3]))])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 1)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Media vs Desviación Estándar')
plt.xlabel('Media')
plt.ylabel('Desviación Estándar')

# Media vs Asimetría
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 1])), yy.ravel(), np.full_like(xx.ravel(), np.mean(X[:,3]))])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 2)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 2], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Media vs Asimetría')
plt.xlabel('Media')
plt.ylabel('Asimetría')

# Media vs Kurtosis
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 1])), np.full_like(xx.ravel(), np.mean(X[:, 2])), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 3)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Media vs Kurtosis')
plt.xlabel('Media')
plt.ylabel('Kurtosis')

# Desviación Estándar vs Asimetría
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), np.mean(X[:,3]))])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 4)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 1], X_test[:, 2], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Desviación Estándar vs Asimetría')
plt.xlabel('Desviación Estándar')
plt.ylabel('Asimetría')

# Desviación Estándar vs Kurtosis
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 2])), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 5)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 1], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Desviación Estándar vs Kurtosis')
plt.xlabel('Desviación Estándar')
plt.ylabel('Kurtosis')

# Asimetría vs Kurtosis
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), np.full_like(xx.ravel(), np.mean(X[:, 1])), xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 3, 6)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title('Asimetría vs Kurtosis')
plt.xlabel('Asimetría')
plt.ylabel('Kurtosis')
plt.tight_layout()
plt.show()
