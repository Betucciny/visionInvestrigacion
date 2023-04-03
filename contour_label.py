import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def find_next(point: tuple, d: int,  imagen: np.array, label_map: np.array) -> (tuple, int):
    delta = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
    y, x = point
    next_point = point
    for i in range(7):
        direction = delta[d]
        next_point = (y + direction[0], x + direction[1])
        if imagen[next_point[0]][next_point[1]] == 0:
            label_map[next_point[0]][next_point[1]] = -1
            d = (d + 1) % 8
        else:
            return next_point, d
    return next_point, d


def trace_contour(start_point: tuple, ds: int, imagen: np.array, label_map: np.array, label: int) -> list:
    next_point, d_next = find_next(start_point, ds, imagen, label_map)
    contour = [next_point]
    current_point = next_point
    done = start_point == next_point
    while not done:
        # print(current_point)
        label_map[current_point[0]][current_point[1]] = label
        d_search = (d_next + 6) % 8
        temp_point, d_next = find_next(current_point, d_search, imagen, label_map)
        previus_point = current_point
        current_point = temp_point
        done = previus_point == start_point and current_point == next_point
        if not done:
            contour.append(current_point)
    print("aaaaaaaaaa")
    return contour


def contour_labeling(imagen: np.array) -> (list, list, np.array):
    Couter = []
    Cinnner = []
    alto, ancho = imagen.shape
    label_map = np.zeros((alto, ancho))
    region_counter = 0
    for y in range(alto):
        current_label = 0
        for x in range(ancho):
            if imagen[y][x] == 1:
                if current_label != 0:
                    label_map[y][x] = current_label
                else:
                    l = label_map[y][x]
                    if l == 0:
                        region_counter += 1
                        l = region_counter
                        conteur = trace_contour((y, x), 0, imagen, label_map, l)
                        Couter.append(conteur)
                        label_map[y][x] = l
            else:
                if current_label != 0:
                    if label_map[y][x] == 0:
                        Cinnner.append(trace_contour((y-1, x), 1, imagen, label_map, l))
                    current_label = 0
    return Couter, Cinnner, label_map


def main():
    imagen = Image.open("img.png")
    imagen = imagen.convert("L")
    imagen = np.where(np.array(imagen) > 128, 0, 1).astype(np.uint8)
    plt.imshow(imagen, cmap="gray")
    plt.show()
    plt.clf()
    # contours_pts = contour_labeling(imagen)
    contours, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_pts = []
    for cnt in contours:
        cnt_pts = []
        for pt in cnt:
            cnt_pts.append(pt[0])
        contours_pts.append(cnt_pts)

    for contours in contours_pts:
        plt.scatter([x[0] for x in contours], [x[1] for x in contours], s=1)

    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
