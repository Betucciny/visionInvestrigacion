import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def flood_filing(imagen: np.array, x: int, y: int) -> np.array:
    list_coords = [(y, x)]
    visitados = []
    alto, ancho = imagen.shape
    img = np.zeros((alto, ancho))
    while list_coords:
        y, x = list_coords.pop()
        if (y, x) in visitados:
            continue
        visitados.append((y, x))
        if 0 <= x < ancho and 0 <= y < alto and imagen[y][x] == 1:
            img[y][x] = 1
            list_coords.append((y + 1, x))
            list_coords.append((y - 1, x))
            list_coords.append((y, x + 1))
            list_coords.append((y, x - 1))
    return img


def region_labeling(imagen: np.array) -> np.array:
    alto, ancho = imagen.shape
    print(alto, ancho)
    regiones = []
    for y in range(alto):
        for x in range(ancho):
            if imagen[y][x] == 1:
                img = flood_filing(imagen, x, y)
                imagen = np.where(img == 1, 0, imagen)
                regiones.append(img)
                print("Region encontrada")
    return regiones


def main():
    imagen = Image.open("img.png")
    imagen = imagen.convert("L")
    imagen = np.where(np.array(imagen) > 128, 0, 1).astype(np.uint8)
    regiones = region_labeling(imagen)
    plt.imshow(imagen, cmap="gray")
    plt.show()
    for i, region in enumerate(regiones):
        plt.imshow(region, cmap="gray")
        plt.title(f"Region {i}")
        plt.show()


if __name__ == "__main__":
    main()
