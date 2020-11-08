import main

def get_bilinear(img, or_pixel):
    (p1, p2, p3, p4) = get_quarter_pixels(or_pixel)

    if is_near_edge(len(img), len(img[0]), [p1, p2, p3, p4]):
        return main.get_nearest_neighbor(img, or_pixel)

    ph = get_horizontal_avg(img, or_pixel, p1, p2)
    pl = get_horizontal_avg(img, or_pixel, p3, p4)

    return get_vertical_avg(or_pixel, p1, p3, ph, pl)

def get_quarter_pixels(or_pixel):
    (horizontal, vertical) = (0, 0)

    if 0.5 > or_pixel[0] - int(or_pixel[0]) > 0:
        vertical = -1
    if 0.5 > or_pixel[1] - int(or_pixel[1]) > 0:
        horizontal = -1

    p1 = (int(or_pixel[0] + vertical), int(or_pixel[1] + horizontal))
    p2 = (int(or_pixel[0] + vertical + 1), int(or_pixel[1] + horizontal))
    p3 = (int(or_pixel[0] + vertical), int(or_pixel[1] + horizontal + 1))
    p4 = (int(or_pixel[0] + vertical + 1), int(or_pixel[1] + horizontal + 1))

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
        if p[0] < 0 or p[0] >= height or p[1] < 0 or p[1] >= width:
            return True
    return False