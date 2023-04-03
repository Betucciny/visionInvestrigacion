from morphology import dilatacion, erosion
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def thinning(imagen: np.array) -> np.array:
    cruz = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    rectan = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    while True:
        imagen = imagen.copy()
        erosionada = erosion(imagen, cruz)
        dilatada = dilatacion(erosionada, cruz)
        diff1 = abs(imagen - dilatada)

        erosionada = erosion(imagen, rectan)
        dilatada = dilatacion(erosionada, rectan)
        diff2 = abs(imagen - dilatada)

        inter = diff1 & diff2
        imagen_new = inter
        if np.array_equal(imagen, imagen_new):
            break
        imagen = imagen_new
    return imagen


def main():
    # Cargar la imagen
    image = np.where(data.horse() > 0, 0, 1).astype(np.uint8)
    skel_image = thinning(image)

    plt.imshow(image, cmap='gray')
    plt.show()
    plt.clf()
    plt.imshow(skel_image, cmap='gray')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
