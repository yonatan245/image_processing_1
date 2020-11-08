import sys

import cv2
import numpy as np


from Matrix import get_matrix

def check_range(pixel, height, width):
    if pixel[0] < 0 or pixel[0] >= height or pixel[1] < 0 or pixel[1] >= width:
        return False
    return True

def get_color(img, i, j, quality, inv_mat):
    pixel = [i, j, 1]
    or_pixel = np.matmul(inv_mat, pixel)

    if not check_range(or_pixel, len(img), len(img[0])):
        return 0
    else:
        if quality == 'N':
            return get_nearest_neighbor(img, or_pixel)
        elif quality == 'B':
            return get_bilinear(img, or_pixel)
        else:
            return get_cubic(img, or_pixel)

def get_nearest_neighbor(img, or_pixel):
    return img[int(or_pixel[0])][int(or_pixel[1])]

def get_bilinear(img, or_pixel):
    (p1, p2, p3, p4) = get_quarter_pixels(or_pixel)

    if is_near_edge(len(img), len(img[0]), [p1, p2, p3, p4]):
        return get_nearest_neighbor(img, or_pixel)

    ph = get_horizontal_avg(img, or_pixel, p1, p2)
    pl = get_horizontal_avg(img, or_pixel, p3, p4)

    return get_vertical_avg(or_pixel, p1, p3, ph, pl)

def get_cubic(img, or_pixel):
    return 0

def get_quarter_pixels(or_pixel):
    (horizontal, vertical) = (0, 0)

    if or_pixel[0] - int(or_pixel[0]) < 0.5:
        vertical = -1
    if or_pixel[1] - int(or_pixel[1]) < 0.5:
        horizontal = -1

    p1 = (or_pixel[0] + vertical, or_pixel[1] + horizontal)
    p2 = (or_pixel[0] + vertical, or_pixel[1] + horizontal + 1)
    p3 = (or_pixel[0] + vertical + 1, or_pixel[1] + horizontal)
    p4 = (or_pixel[0] + vertical + 1, or_pixel[1] + horizontal + 1)

    return p1, p2, p3, p4

def get_horizontal_avg(img, or_pixel, p1, p2):
    w1 = p2[0] - or_pixel[0]
    w2 = or_pixel[0] - p1[0]

    return w1 * img[p1] + w2 * img[p2]

def get_vertical_avg(or_pixel, p1, p3, pl, ph):
    w1 = p3[1] - or_pixel[1]
    w2 = or_pixel[1] - p1[1]

    return w1 * pl + w2 * ph

def is_near_edge(height, width, points):
    for p in points:
        if p[0] < 0 or p[0] > height or p[1] < 0 or p[1] > width:
            return True
    return False


if __name__ == '__main__':

    img_path = sys.argv[1]
    txt_path = sys.argv[2]
    quality = sys.argv[3]

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    mat = get_matrix(txt_path)
    inv_mat = np.linalg.inv(mat)

    # ang = 34 * np.pi / 180
    #
    # mat = np.array([[2*np.cos(ang), np.sin(ang), 0],
    #                 [-np.sin(ang), 2*np.cos(ang), 0],
    #                 [20, 50, 1]])

    # mat = np.array([[2, 0, 0],
    #                 [0, 2, 0],
    #                 [200, 500, 1]])

    trans_h = int(mat[2, 0])
    trans_w = int(mat[2, 1])

    p1 = np.matmul(mat, [0,0,1])
    p2 = np.matmul(mat, [len(img),0,1])
    p3 = np.matmul(mat, [0,len(img[0]),1])
    p4 = np.matmul(mat, [len(img),len(img[0]),1])

    min_h = int(min(p1[0], p2[0], p3[0], p4[0]))
    max_h = int(max(p1[0], p2[0], p3[0], p4[0]))
    min_w = int(min(p1[1], p2[1], p3[1], p4[1]))
    max_w = int(max(p1[1], p2[1], p3[1], p4[1]))

    h_off = np.abs(min(0, min_h))
    w_off = np.abs(min(0, min_w))

    new_img = np.zeros((max_h - min_h + trans_h, max_w - min_w + trans_w), dtype=int)

    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            new_img[i, j] = int(get_color(img, i - h_off - trans_h, j - w_off - trans_w, quality, inv_mat))


            # pixel = [i - h_off - trans_h, j - w_off - trans_w, 1]
            # or_pixel = np.matmul(inv_mat, pixel)
            #
            # if not check_range(or_pixel, len(img), len(img[0])):
            #     new_img[i, j] = 0
            # else:
            #     new_img[i, j] = img[int(or_pixel[0])][int(or_pixel[1])]



            # or_pixel[0] = int(min(or_pixel[0], len(img)-1))
            # or_pixel[1] = int(min(or_pixel[1], len(img[0])-1))

            # print(or_pixel)
            # new_img[i][j] = int(img[int(np.floor(or_pixel[0]))][int(np.floor(or_pixel[1]))])
            # new_img[i][j] = img[int(np.floor(or_pixel[0]))][int(np.floor(or_pixel[1]))]




    # print(mat)
    # print(img)
    print(new_img)
    cv2.imwrite('new_img.png', new_img)
    cv2.imshow('ziv', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()