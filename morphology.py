from PIL import Image
import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt


def generate_imagen():
    circle_image = np.zeros((300, 500))
    circle_image[disk((150, 125), 90)] = 1
    circle_image[disk((150, 325), 90)] = 1
    for x in range(200):
        circle_image[np.random.randint(300), np.random.randint(500)] = 1
    for x in range(200):
        circle_image[np.random.randint(300), np.random.randint(500)] = 0
    return circle_image


def binarizacion(imagen: np.array, umbral: int) -> np.array:
    alto, ancho = imagen.shape
    imagen_binarizada = np.zeros((alto, ancho), dtype=np.uint8)
    for y in range(alto):
        for x in range(ancho):
            if imagen[y][x] > umbral:
                imagen_binarizada[y][x] = 1
    return imagen_binarizada


def dilatacion(imagen: np.array, kernel: np.array) -> np.array:
    alto, ancho = imagen.shape
    imagen_dilatada = np.zeros((alto, ancho), dtype=np.uint8)
    kernel_alto, kernel_ancho = kernel.shape
    if kernel_alto % 2 == 0 or kernel_ancho % 2 == 0:
        raise ValueError("El kernel debe ser de tamaño impar")
    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            for ky in range(-kernel_alto//2, kernel_alto//2 + 1):
                for kx in range(-kernel_ancho//2, kernel_ancho//2 + 1):
                    if kernel[ky][kx] == 1 and imagen[y + ky][x + kx] == 1:
                        imagen_dilatada[y][x] = 1
                        break

    return imagen_dilatada


def erosion(imagen: np.array, kernel: np.array) -> np.array:
    alto, ancho = imagen.shape
    imagen_erosionada = np.zeros((alto, ancho), dtype=np.uint8)
    kernel_alto, kernel_ancho = kernel.shape
    if kernel_alto % 2 == 0 or kernel_ancho % 2 == 0:
        raise ValueError("El kernel debe ser de tamaño impar")
    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            for ky in range(-kernel_alto//2, kernel_alto//2 + 1):
                for kx in range(-kernel_ancho//2, kernel_ancho//2 + 1):
                    if kernel[ky][kx] == 1 and imagen[y + ky][x + kx] == 0:
                        break
                else:
                    continue
                break
            else:
                imagen_erosionada[y][x] = 1
    return imagen_erosionada


def apertura(imagen: np.array, kernel: np.array) -> np.array:
    imagen_erosionada = erosion(imagen, kernel)
    imagen_dilatada = dilatacion(imagen_erosionada, kernel)
    return imagen_dilatada


def cierre(imagen: np.array, kernel: np.array) -> np.array:
    imagen_dilatada = dilatacion(imagen, kernel)
    imagen_erosionada = erosion(imagen_dilatada, kernel)
    return imagen_erosionada


def main():
    imagen_original = generate_imagen()
    imagen_original = np.array(imagen_original, dtype=np.uint8)
    # imagen_binarizada = binarizacion(imagen_original, 128)
    imagen_binarizada = imagen_original
    imagen_dilatada = dilatacion(np.array(imagen_binarizada), np.array([[0,1,0], [1,1,1], [0,1,0]]))
    imagen_erosionada = erosion(np.array(imagen_binarizada), np.array([[0,1,0], [1,1,1], [0,1,0]]))
    imagen_apertura = apertura(np.array(imagen_binarizada), np.array([[0,1,0], [1,1,1], [0,1,0]]))
    imagen_cierre = cierre(np.array(imagen_binarizada), np.array([[0,1,0], [1,1,1], [0,1,0]]))
    plt.imshow(imagen_original, cmap="binary")
    plt.title("Imagen original")
    plt.show()
    plt.clf()
    plt.imshow(imagen_dilatada, cmap="binary")
    plt.title("Imagen dilatada")
    plt.show()
    plt.clf()
    plt.imshow(imagen_erosionada, cmap="binary")
    plt.title("Imagen erosionada")
    plt.show()
    plt.clf()
    plt.imshow(imagen_apertura, cmap="binary")
    plt.title("Imagen apertura")
    plt.show()
    plt.clf()
    plt.imshow(imagen_cierre, cmap="binary")
    plt.title("Imagen cierre")
    plt.show()
    plt.clf()


if __name__ == "__main__":
   main()
