import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def find_next(point: tuple, d: int,  imagen: np.array, label_map: np.arrat) -> (tuple, int):
    delta = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    y, x = point
    next_point = point
    for i in range(7):
        direction = delta[d]
        next_point = (y + direction[0], x + direction[1])
        if imagen[next_point[0]][next_point[1]]:
            label_map[next_point[0]][next_point[1]] = -1
            d = (d + 1) % 8
        else:
            return next_point, d
    return next_point, d


def trace_contour(start_point: tuple, ds: int, imagen: np.array, label_map: np.array, label: int  ) -> list:
    next_point, d_next = find_next(start_point, ds, imagen, label_map)
    contour = [next_point]
    previus_point = start_point
    current_point = next_point
    while start_point != next_point:
        label_map[current_point[0]][current_point[1]] = label
        d_search = (d_next + 6) % 8
        temp_point, d_next = find_next(current_point, d_search, imagen, label_map)
        previus_point = current_point
        current_point = temp_point
        done = previus_point == start_point and current_point == next_point
        if not done:
            contour.append(current_point)
    return contour


def