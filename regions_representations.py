import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sequential_segmentation import region_labeling
from ZhangSuen import neighbours


def matrix2rle(matrix: np.array) -> np.array:
    rle = []
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        start = None
        count = 0
        for j in range(row.size):
            if row[j] == 1:
                if start is None:
                    start = j
                count += 1
            elif count > 0:
                rle.append((i, start, count))
                start = None
                count = 0
        if count > 0:
            rle.append([i, start, count])
    return np.asarray(rle)


def main():
    imagen = Image.open("img.png")
    imagen = imagen.convert("L")
    imagen = np.where(np.array(imagen) > 128, 0, 1).astype(np.uint8)
    regiones = region_labeling(imagen)
    rle = []
    for region in regiones:
        rle.append(matrix2rle(region))
    for i, region in enumerate(regiones):
        print(f"Region {i} es representada por la siguiente matriz:\n {region}")
        print(f"Region {i} es representada por la siguiente RLE:\n {rle[i]}")


if __name__ == "__main__":
    main()
