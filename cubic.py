import numpy as np

import main


def get_cubic(img, or_pixel):
    pixels = get_quarter_pixels(or_pixel)

    if is_near_edge(len(img), len(img[0]), pixels):
        return main.get_nearest_neighbor(img, or_pixel)

    output = 0

    for i in range(4):
        for j in range(4):
            output += img[pixels[i, j]] * get_weight(or_pixel, pixels[i, j])

    return max(min(output, 255), 0)



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

def get_distance(src, target):
    d = abs(src - target)

    if d < 1:
        return 1.5 * (d**3) - 2.5 * (d**2) + 1
    if d < 2:
        return -0.5 * (d**3) + 2.5 * (d**2) - 4*d +2
    return 0

def get_weight(pixel, target):
    horizontal = get_distance(pixel[0], target[0])
    vertical = get_distance(pixel[1], target[1])

    return horizontal * vertical

#
def is_near_edge(height, width, points):
    for line in points:
        for p in line:
            if p[0] < 0 or p[0] >= height or p[1] < 0 or p[1] >= width:
                return True
    return False