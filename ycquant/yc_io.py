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
    diff = price_table_lag[:len(price_table)] - price_table

    binary_vector = diff[diff > 0]
    pass


def read_unsupervised_data(data_path):
    """
    Reading x_data and price-table which is not the target of the algorithm
    :param data_path: str
        data-table path
    :return:
    """
    df = pd.read_csv(data_path)
    x_data = df.drop("price", axis=1)
    price = df["price"]

    return x_data.as_matrix(), price.as_matrix()


def read_classification_data(data_path):
    """
    Reading x_data and price-table which is not the target of the algorithm
    :param data_path: str
        data-table path
    :return:
    """
    x_data, price = read_unsupervised_data(data_path)

    return x_data, price


def save_data_to(obj, file):
    pass
