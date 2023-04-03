import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from regions_representations import matrix2acc, matrix2rle
from regions_finding import region_labeling


def perimeter(acc_contorno: list) -> int:
    return sum([1 if i % 2 == 0 else sqrt(2) for i in acc_contorno])


def areaContour(contorno: list) -> int:
    M = len(list)
    A = 0
    for i in range(M):
        punto1 = contorno[i][0] * contorno[(i+1) % M][1]
        punto2 = contorno[i][1] * contorno[(i+1) % M][0]
        A += punto1 - punto2
    return abs(A / 2)

def areaMatrix(contorno: np.array) -> int:
    return np.sum(contorno)


def circularity(area: int, perimeter: int) -> float:
    return 4 * np.pi * area / perimeter ** 2


def bounding_box(contorno: list) -> tuple:
    x = [i[0] for i in contorno]
    y = [i[1] for i in contorno]
    return min(x), min(y), max(x), max(y)


def centroid(region: np.array) -> tuple:
    list_region = []
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            if region[i, j] == 1:
                list_region.append((i, j))
    x = [i[0] for i in list_region]
    y = [i[1] for i in list_region]
    return np.mean(x), np.mean(y)


def central_moments(region: np.array, p: int, q: int) -> float:
    x, y = centroid(region)
    M = 0
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            if region[i, j] == 1:
                M += (i - x) ** p * (j - y) ** q
    return M


def normal_central_moment(region: np.array, p: int, q: int) -> float:
    M = central_moments(region, p, q)
    M_00 = central_moments(region, 0, 0)
    return M / M_00 ** ((p + q) / 2 + 1)



