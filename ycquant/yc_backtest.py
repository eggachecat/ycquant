import numpy as np


class FitCurveStrategy:
    def __init__(self):
        self.__createform = 'python'

    @staticmethod
    def get_profit_by_op(y_pred, y_true, indicies=None):
        return y_pred

    @staticmethod
    def get_reward(y_pred, y_true, indicies=None, debug=False):

        if indicies is not None:
            y_true = y_true[indicies]
            y_pred = y_pred[indicies]

        return np.abs(y_pred - y_true).mean()

    @staticmethod
    def get_info(prediction, price_table, indicies=None):
        if indicies is not None:
            price_table = price_table[indicies]
            prediction = prediction[indicies]
        y_pred = FitCurveStrategy.get_op_arr(prediction)

        return FitCurveStrategy.get_info_by_op(y_pred, price_table, indicies)

    @staticmethod
    def get_info_by_op(op_arr, price_table, indicies=None):
        if indicies is not None:
            price_table = price_table[indicies]
            op_arr = op_arr[indicies]
        return {
            "profit_arr": FitCurveStrategy.get_profit_by_op(op_arr, price_table),
            "op_arr": op_arr
        }

    @staticmethod
    def get_op_arr(value_list):
        return value_list


class CrossBarStrategy:
    def __init__(self):
        self.__createform = 'python'

    @staticmethod
    def get_profit_by_op(op_arr, price_table, indicies=None):

        if indicies is not None:
            price_table = price_table[indicies]
            op_arr = op_arr[indicies]

        investment = -1 * op_arr * price_table

        # if not investment.size % 2 == 0:
        #     investment[-1] = 0

        transcation_indicies = np.nonzero(investment)
        transcation = investment[transcation_indicies]

        if not transcation.size % 2 == 0:
            transcation[-1] = transcation[-2]

        if transcation.size > 0:

            transcation_lag = np.roll(transcation, 1)

            _profit_arr = transcation + transcation_lag
            _profit_arr[0] = 0

            profit_arr = np.zeros(len(op_arr))
            profit_arr[transcation_indicies] = _profit_arr

        else:
            profit_arr = np.zeros(len(op_arr))
        # print(profit_arr)
        return profit_arr

    @staticmethod
    def get_reward(prediction, price_table, indicies=None, debug=False):

        if indicies is not None:
            prediction = prediction[indicies]
            price_table = price_table[indicies]


        op_arr = CrossBarStrategy.get_op_arr(prediction)
        profit_arr = CrossBarStrategy.get_profit_by_op(op_arr, price_table)

        if debug:
            print("prediction", op_arr)
            print("op_arr", op_arr)
            print("profit_arr", profit_arr)

        return np.sum(profit_arr)

    @staticmethod
    def price_to_movement(price_pred):
        future_price_table = np.roll(price_pred, -1)
        y_pred = future_price_table - price_pred
        y_pred[-1] = 0
        y_pred[y_pred == 0] = 1

        return y_pred

    @staticmethod
    def get_info(prediction, price_table, indicies=None):
        if indicies is not None:
            price_table = price_table[indicies]
            prediction = prediction[indicies]
        op_arr = CrossBarStrategy.get_op_arr(prediction)

        return CrossBarStrategy.get_info_by_op(op_arr, price_table, indicies)

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
