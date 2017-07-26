import numpy as np


class CrossBarStrategy:
    def __init__(self):
        self.__createform = 'python'

    @staticmethod
    def get_profit_by_op(op_arr, price_table, indicies=None):

        if indicies is not None:
            price_table = price_table[indicies]
            op_arr = op_arr[indicies]

        investment = -1 * op_arr * price_table
        transcation_indicies = np.nonzero(investment)

        if len(transcation_indicies) > 0:

            transcation = investment[transcation_indicies]
            transcation_lag = np.roll(transcation, 1)

            _profit_arr = transcation + transcation_lag

            _profit_arr[0] = 0

            profit_arr = np.zeros(len(op_arr))
            profit_arr[transcation_indicies] = _profit_arr

        else:
            profit_arr = np.zeros(len(op_arr))

        return profit_arr

    @staticmethod
    def get_reward(prediction, price_table, indicies=None, debug=False):

        if indicies is not None:
            price_table = price_table[indicies]
            prediction = prediction[indicies]

        op_arr = CrossBarStrategy.get_op_arr(prediction)
        profit_arr = CrossBarStrategy.get_profit_by_op(op_arr, price_table)

        if debug:
            print("prediction", op_arr)
            print("op_arr", op_arr)
            print("profit_arr", profit_arr)

        return np.sum(profit_arr)

    @staticmethod
    def get_info(prediction, price_table, indicies=None):
        if indicies is not None:
            price_table = price_table[indicies]
            prediction = prediction[indicies]

        op_arr = CrossBarStrategy.get_op_arr(prediction)
        profit_arr = CrossBarStrategy.get_profit_by_op(op_arr, price_table)
        return {
            "profit_arr": profit_arr,
            "op_arr": op_arr
        }

    @staticmethod
    def get_info_by_op(op_arr, price_table, indicies=None):
        if indicies is not None:
            price_table = price_table[indicies]
            op_arr = op_arr[indicies]
        return {
            "profit_arr": CrossBarStrategy.get_profit_by_op(op_arr, price_table),
            "op_arr": op_arr
        }

    @staticmethod
    def get_op_arr(value_list):

        value_list_lag = np.roll(value_list, 1)
        value_list_lag[0] = 0

        flag = value_list * value_list_lag < 0
        flag = flag.astype(int)

        return np.sign(flag * value_list)
