import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def neighborhood(imagen: np.array, x: int, y: int) -> list:
    alto, ancho = imagen.shape
    vecinos = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= ancho:
                continue
            if y + j < 0 or y + j >= alto:
                continue
            vecinos.append(imagen[y + j][x + i])
    return vecinos


def assign_initial(imagen: np.array) -> np.array:
    imagen = imagen.copy()
    alto, ancho = imagen.shape
    C = set()
    actual = 2
    for y in range(alto):
        for x in range(ancho):
            if imagen[y][x] == 1:
                neighbors = neighborhood(imagen, x, y)
                if all(val == 0 or val == 1 for val in neighbors):
                    imagen[y][x] = actual
                    actual += 1
                elif neighbors.count(1) + neighbors.count(0) == len(neighbors) - 1:
                    imagen[y][x] = max(neighbors)
                else:
                    etiqueta = min([val for val in neighbors if val != 0 and val != 1])
                    imagen[y][x] = etiqueta
                    conflictos = list(set([val for val in neighbors if val != 0 and val != 1 and val != etiqueta]))
                    for conflicto in conflictos:
                        C.add((etiqueta, conflicto))
    return imagen, C, actual


def resolve_conflicts(imagen: np.array, C: set, m: int) -> np.array:
    R = [{i} for i in range(2, m)]
    for conflictos in C:
        seta = set()
        setb = set()
        for setR in R:
            if conflictos[0] in setR:
                seta = setR
            if conflictos[1] in setR:
                setb = setR
        if seta != setb:
            R.remove(seta)
            R.remove(setb)
            R.append(seta.union(setb))
    for conflicto in R:
        for i in conflicto:
            imagen = np.where(imagen == i, min(conflicto), imagen)
    return imagen



def sequential_labeling(imagen: np.array) -> np.array:
    imagen_new, C, m = assign_initial(imagen)
    imagen_new = resolve_conflicts(imagen_new, C, m)
    return imagen_new


def region_labeling(imagen: np.array) -> list:
    regiones = []
    imagen = sequential_labeling(imagen)
    for i in range(2, imagen.max() + 1):
        region = np.where(imagen == i, 1, 0)
        if region.sum() > 0:
            regiones.append(region)
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
