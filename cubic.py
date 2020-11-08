import numpy as np

import main


def get_cubic(img, or_pixel):
    pixels = get_quarter_pixels(or_pixel)

    if is_near_edge(len(img), len(img[0]), pixels):
        return main.get_nearest_neighbor(img, or_pixel)

    p0 = get_horizontal_avg(img, or_pixel, pixels[0][0], pixels[0][1], pixels[0][2], pixels[0][3])
    p1 = get_horizontal_avg(img, or_pixel, pixels[1][0], pixels[1][1], pixels[1][2], pixels[1][3])
    p2 = get_horizontal_avg(img, or_pixel, pixels[2][0], pixels[2][1], pixels[2][2], pixels[2][3])
    p3 = get_horizontal_avg(img, or_pixel, pixels[3][0], pixels[3][1], pixels[3][2], pixels[3][3])

    output = get_vertical_avg(or_pixel, p0, p1, p2, p3)

    return output


def get_quarter_pixels(or_pixel):
    (horizontal, vertical) = (0, 0)

    if 0.5 > or_pixel[0] - int(or_pixel[0]) > 0:
        vertical = -1
    if 0.5 > or_pixel[1] - int(or_pixel[1]) > 0:
        horizontal = -1

    pixels = np.empty((4, 4), dtype=object)

    for i in range(0, 4):
        for j in range(0, 4):
            pixels[i, j] = (int(or_pixel[0] + vertical + i - 1),
                            int(or_pixel[1] + horizontal + j - 1))

    return pixels


def get_horizontal_avg(img, or_pixel, p0, p1, p2, p3):
    a = -0.5*img[p0] + 1.5*img[p1] - 1.5*img[p2] + 0.5*img[p3]
    b = img[p0] - 2.5*img[p1] + 2*img[p2] - 0.5*img[p3]
    c = -0.5 * img[p0] + 0.5*img[p2]
    d = img[p1]

    output = a*np.power(or_pixel[0], 3) + b*np.power(or_pixel[0], 2) + c*or_pixel[0] + d

    return output


def get_vertical_avg(or_pixel, p0, p1, p2, p3):
    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    d = p1

    return a * np.power(or_pixel[1], 3) + b * np.power(or_pixel[1], 2) + c * or_pixel[1] + d


def is_near_edge(height, width, points):
    for line in points:
        for p in line:
            if p[0] < 0 or p[0] >= height or p[1] < 0 or p[1] >= width:
                return True
    return False
