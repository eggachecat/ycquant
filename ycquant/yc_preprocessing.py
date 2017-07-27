from ycquant import yc_io
import pandas as pd


def split_train_and_test(file_path, header=None, sep=',', split_ratio=0.8, data_projection_list=None):
    df = pd.read_csv(file_path, header=header, sep=sep)

    if data_projection_list is None:
        data_projection_list = [i for i in range(56)]
        data_projection_list.append(len(df.columns) - 1)
    print(data_projection_list)
    flag = int(split_ratio * len(df.index))
    df_projection = df[data_projection_list]

    train_file_name, test_file_name = file_path + ".train", file_path + ".test"
    df_projection[:flag].to_csv(path_or_buf=train_file_name, index=False, header=None, sep=',')
    df_projection[flag:].to_csv(path_or_buf=test_file_name, index=False, header=None, sep=',')

    return train_file_name, test_file_name
