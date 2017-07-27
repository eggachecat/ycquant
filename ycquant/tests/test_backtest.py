from ycquant.yc_backtest import *
import numpy as np
import unittest
import pandas as pd
import os

script_dir = os.path.dirname(__file__)


class TestCrossBarStrategy(unittest.TestCase):
    # def test_get_op_arr(self):
    #     file_path = "data/test_backtest"
    #
    #     abs_file_path = os.path.join(script_dir, file_path)
    #     df = pd.read_csv(abs_file_path, header=None, sep="\t")
    #
    #     value_list = df[df.columns[0]].as_matrix()
    #     op_arr_true = df[df.columns[2]].as_matrix()
    #
    #     op_arr_pred = CrossBarStrategy.get_op_arr(value_list)
    #
    #     assert np.array_equal(op_arr_true, op_arr_pred), "Error"

    def test_get_reward(self):
        file_path = "data/test_backtest"

        abs_file_path = os.path.join(script_dir, file_path)
        df = pd.read_csv(abs_file_path, header=None, sep="\t")

        value_list = df[df.columns[0]].as_matrix()
        price_table = df[df.columns[1]].as_matrix()
        op_arr_true = df[df.columns[2]].as_matrix()

        profit_arr = CrossBarStrategy.get_profit_by_op(op_arr_true, price_table)
        profit = CrossBarStrategy.get_reward(value_list, price_table)

        print(profit_arr, profit)


if __name__ == '__main__':
    unittest.main()
