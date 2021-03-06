import cv2
import numpy as np

nombre_imagen = 'hoja3.jpg'
Im1 = cv2.imread(nombre_imagen)

# ------------------------------------------------------------------
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Elegimos el umbral de verde en HSV
umbral_bajo = (0, 0, 0)
umbral_alto = (210, 210, 210)
# hacemos la mask y filtramos en la original
mask = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)
res = cv2.bitwise_and(img, img, mask=mask)

# ----------------------------Esquinas de harris------------------------------------


img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(img, 3, 3, 0.05)

# replicamos matriz en eje Z para que sea de 3 canales
img_out = img[:, :, np.newaxis]
img_out = np.tile(img_out, (1, 1, 3))
img_out[dst > (0.01 * dst.max())] = [255, 0, 0]
mask = img * 0
mask[dst > (0.01 * dst.max())] = 255


# --------------------------Encontrar esquinas de la hoja--------------------------------------
def calc_distancia(inicio, meta):
    return round(((inicio[0] - meta[0]) ** 2 + (inicio[1] - meta[1]) ** 2) ** 0.5, 2)


filas, columnas = mask.shape

arrCoor = np.where(mask != 0)

arr = []
for i in range(len(arrCoor[0])):
    arr.append((arrCoor[1][i], arrCoor[0][i]))

pixel_izq_arr = (0, 0)
pixel_der_arr = (columnas - 1, 0)
pixel_izq_aba = (0, filas - 1)
pixel_der_aba = (columnas - 1, filas - 1)

izq_arr = []
for pix in arr:
    izq_arr.append((calc_distancia(pix, pixel_izq_arr), pix))

der_arr = []
for pix in arr:
    der_arr.append((calc_distancia(pix, pixel_der_arr), pix))

izq_aba = []
for pix in arr:
    izq_aba.append((calc_distancia(pix, pixel_izq_aba), pix))

der_aba = []
for pix in arr:
    der_aba.append((calc_distancia(pix, pixel_der_aba), pix))

izq_arr = sorted(izq_arr)
der_arr = sorted(der_arr)
izq_aba = sorted(izq_aba)
der_aba = sorted(der_aba)

coor_izq_arr = izq_arr[1][1]
coor_der_arr = der_arr[1][1]
coor_izq_aba = izq_aba[1][1]
coor_der_aba = der_aba[1][1]

# ----------------------Perspectiva-------------------------------------------------


pts1 = np.float32(
    [[coor_izq_arr[0], coor_izq_arr[1]], [coor_der_arr[0], coor_der_arr[1]], [coor_izq_aba[0], coor_izq_aba[1]],
     [coor_der_aba[0], coor_der_aba[1]]])
pts2 = np.float32([[0, 0], [348, 0], [0, 463], [348, 463]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(Im1, M, (348, 463))

cv2.imwrite('resultados/resultado-' + nombre_imagen, dst)
cv2.imwrite('resultados/mask-' + nombre_imagen, mask)
