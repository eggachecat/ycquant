import pandas as pd
import numpy as np


def convert_price_to_binary(price_table):
    """

    :param price_table: float-list

    :return:
        A binary-list which is made by doing substraction between two neighbour price in price_table
        Definantion:
            1 for price going up
            0 for price going down
    """

    price_table_lag = np.insert(price_table, [0], price_table[0])
    # src:  [p_1, p_2, p_3]
    # lag:  [p_1, p_1, p_2]
    #       [0, p_2-p_1, p_3-p_2]

    binary_vector = price_table - price_table_lag[:len(price_table)]

    binary_vector[binary_vector > 0] = 1
    binary_vector[binary_vector < 0] = 0

    return binary_vector[1:].astype(int)


def read_unsupervised_data(data_path, header=None, sep=','):
    """
    Reading x_data and price-table which is not the target of the algorithm
    :param data_path: str
        data-table path
    :return:
    """
    df = pd.read_csv(data_path, header=header, sep=sep)

    if not header is None:
        x_data = df.drop("price", axis=1)
        price = df["price"]

    else:
        x_data = df.drop(df.columns[len(df.columns) - 1], axis=1)
        price = df[df.columns[len(df.columns) - 1]]

    return x_data.as_matrix().astype(float), price.as_matrix().astype(float)


def read_classification_data(data_path):
    """
    Reading x_data and price-table which is not the target of the algorithm
    :param data_path: str
        data-table path
    :return:
    """
    x_data, price = read_unsupervised_data(data_path)

    return x_data[:len(price) - 1], convert_price_to_binary(price)


def save_data_to(obj, file):
    pass
