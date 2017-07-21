from ycquant import yc_io
import pandas as pd


def split_train_and_test(file_path, split_ratio=0.8):
    df = pd.read_csv(file_path)
    flag = int(split_ratio * len(df.index))

    df[:flag].to_csv(path_or_buf=file_path + ".train", index=False)
    df[flag:].to_csv(path_or_buf=file_path + ".test", index=False)


