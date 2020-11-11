import sys
import cv2
import numpy as np
import bilinear as bi
import cubic as cu


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
            return bi.get_bilinear(img, or_pixel)
        else:
            return cu.get_cubic(img, or_pixel)

def get_nearest_neighbor(img, or_pixel):
    return img[int(or_pixel[0])][int(or_pixel[1])]


if __name__ == '__main__':
    img_path = sys.argv[1]
    txt_path = sys.argv[2]
    quality = sys.argv[3]

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    mat = get_matrix(txt_path)
    inv_mat = np.linalg.inv(mat)

    trans_w = int(mat[2, 0])
    trans_h = int(mat[2, 1])

    p1 = np.matmul(mat, [0, 0, 1])
    p2 = np.matmul(mat, [len(img), 0, 1])
    p3 = np.matmul(mat, [0, len(img[0]), 1])
    p4 = np.matmul(mat, [len(img), len(img[0]), 1])

    min_h = int(min(p1[0], p2[0], p3[0], p4[0]))
    max_h = int(max(p1[0], p2[0], p3[0], p4[0]))
    min_w = int(min(p1[1], p2[1], p3[1], p4[1]))
    max_w = int(max(p1[1], p2[1], p3[1], p4[1]))

    h_off = np.abs(min(0, min_h))
    w_off = np.abs(min(0, min_w))

    new_img = np.zeros((max_h - min_h + abs(trans_h), max_w - min_w + abs(trans_w)), dtype=np.float)

    if trans_h < 0:
        # h_off += trans_h
        trans_h = 0

    if trans_w < 0:
        # w_off += trans_w
        trans_w = 0

    for i in range(len(new_img)):
        for j in range(len(new_img[0])):

            # if 0 <= i + trans_h < len(new_img) and 0 <= j + trans_w < len(new_img[0]):
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
    cv2.imwrite('new_img.png', new_img)
    # cv2.imshow(new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()