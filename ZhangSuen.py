from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data


def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    Image_Thinned = image.copy()
    changing1 = changing2 = 1
    while changing1 or changing2:
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and
                        2 <= sum(n) <= 6 and
                        transitions(n) == 1 and
                        P2 * P4 * P6 == 0 and
                        P4 * P6 * P8 == 0):
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and
                        2 <= sum(n) <= 6 and
                        transitions(n) == 1 and
                        P2 * P4 * P8 == 0 and
                        P2 * P6 * P8 == 0):
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


def invert_image(image):
    return np.where(image > 0, 0, 1).astype(np.uint8)


def main():
    # Cargar la imagen
    image = invert_image(data.horse())
    image = np.array(image)

    # Aplicar el algoritmo de Zhang-Suen
    skel_image = zhangSuen(image)

    # Mostrar la imagen original y la imagen esquelética
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.clf()
    plt.imshow(skel_image, cmap='gray')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
