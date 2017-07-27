import numpy as np
import pandas as pd


def target_func(row):
    return [row[0] * row[1] + row[2] * row[2] + row[3] * row[3] + row[4]]


def make_target(x_matrix):
    y = []
    for row in x_matrix:
        y.append(target_func(row))

    return np.array(y)


def main(file_path):
    print(file_path)
    n_dim = 5
    n_data = 500
    rand_matrix = np.random.rand(n_data, n_dim)

    df = pd.DataFrame(data=np.hstack((
        rand_matrix, make_target(rand_matrix)
    )))
    df.to_csv(file_path, index=False, header=None)


main("../../data/random_curve.csv")
