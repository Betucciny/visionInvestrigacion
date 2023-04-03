import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sequential_segmentation import region_labeling
import cv2


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


def matrix2acc(contorno: list[list]) -> list:
    code = {
        (0, 1): 0,
        (1, 1): 1,
        (1, 0): 2,
        (1, -1): 3,
        (0, -1): 4,
        (-1, -1): 5,
        (-1, 0): 6,
        (-1, 1): 7
    }
    acc_contorno = []
    for i in range(len(contorno)):
        if i == len(contorno) - 1:
            pt1 = contorno[i]
            pt2 = contorno[0]
            dif = (pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dif in code:
                acc_contorno.append(code[dif])
        else:
            pt1 = contorno[i]
            pt2 = contorno[(i + 1)]
            dif = (pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dif in code:
                acc_contorno.append(code[dif])
    return acc_contorno


def main():
    imagen = Image.open("img.png")
    imagen = imagen.convert("L")
    imagen = np.where(np.array(imagen) > 128, 0, 1).astype(np.uint8)
    regiones = region_labeling(imagen)

    rle = []
    acc = []
    for region in regiones:
        temp = np.where(region == 1, 255, 0).astype(np.uint8)
        contour, _ = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_pts = []
        for cnt in contour:
            cnt_pts = []
            for pt in cnt:
                cnt_pts.append(pt[0])
            contours_pts.append(cnt_pts)
        acc_contorno = []
        for i, contorno in enumerate(contours_pts):
            acc_contorno.append(matrix2acc(contorno))
        acc.append(acc_contorno)
        rle.append(matrix2rle(region))

    for i, region in enumerate(regiones):
        plt.imshow(region, cmap="gray")
        plt.title(f"Region {i}")
        plt.show()
        print(f"Region {i} es representada por la siguiente matriz:\n {region}")
        print(f"Region {i} es representada por la siguiente RLE:\n {rle[i]}")
        print(f"Region {i} es representada por los siguientes ACC:\n {acc[i]}")




if __name__ == "__main__":
    main()
