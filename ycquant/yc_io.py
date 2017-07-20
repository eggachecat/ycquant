import pandas as pd


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
