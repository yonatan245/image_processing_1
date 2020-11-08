import cv2
import numpy as np


from Matrix import get_matrix

def check_range(pixel, height, width):
    if pixel[0] < 0 or pixel[0] >= height or pixel[1] < 0 or pixel[1] >= width:
        return False
    return True


if __name__ == '__main__':

    # print(get_matrix('text.txt'))

    img = cv2.cvtColor(cv2.imread('pic1.jpg'), cv2.COLOR_BGR2GRAY)

    ang = -120 * np.pi / 180.0

    # mat = np.array([[2*np.cos(ang), np.sin(ang), 0],
    #                 [-np.sin(ang), 2*np.cos(ang), 0],
    #                 [20, 50, 1]])

    # mat = np.array([[2, 0, 0],
    #                 [0, 2, 0],
    #                 [200, 500, 1]])

    mat = get_matrix("text.txt")

    inv_mat = np.linalg.inv(mat)

    print(inv_mat)

    trans_h = mat[2,0]
    trans_w = mat[2,1]

    p1 = np.matmul(mat, [0,0,1])
    p2 = np.matmul(mat, [len(img),0,1])
    p3 = np.matmul(mat, [0,len(img[0]),1])
    p4 = np.matmul(mat, [len(img),len(img[0]),1])

    print(p1)
    print(p2)
    print(p3)
    print(p4)

    min_h = int(min(p1[0], p2[0], p3[0], p4[0]))
    max_h = int(max(p1[0], p2[0], p3[0], p4[0]))
    min_w = int(min(p1[1], p2[1], p3[1], p4[1]))
    max_w = int(max(p1[1], p2[1], p3[1], p4[1]))

    h_off = np.abs(min(0, min_h))
    w_off = np.abs(min(0, min_w))

    print('h_off = ', h_off)
    print('w_off = ', w_off)

    new_img = np.zeros((max_h - min_h, max_w - min_w), dtype=int)
    print(new_img.shape)

    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            pixel = [i - h_off - trans_h, j - w_off - trans_w, 1]
            or_pixel = np.matmul(inv_mat, pixel)

            if not check_range(or_pixel, len(img), len(img[0])):
                new_img[i, j] = 0
            else:
                new_img[i, j] = img[int(or_pixel[0])][int(or_pixel[1])]



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