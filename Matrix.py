import numpy as np

def get_matrix(path):

    file = open(path, "r")
    mat = np.eye(3)

    lines = file.readlines()
    for ln in lines:
        line = ln.split()
        new_mat = np.eye(3)
        # print(line[1])

        if(line[0] == 'S'):
            new_mat[0][0] = float(line[1])
            new_mat[1][1] = float(line[2])

        if(line[0] == 'T'):
            new_mat[2][0] = int(line[1])
            new_mat[2][1] = int(line[2])

        if(line[0] == 'R'):
            ang = float(line[1]) * np.pi / 180.0
            new_mat[0][0] = np.cos(ang)
            new_mat[0][1] = np.sin(ang)
            new_mat[1][0] = -np.sin(ang)
            new_mat[1][1] = np.cos(ang)

        mat = np.matmul(mat, new_mat)

    return mat

