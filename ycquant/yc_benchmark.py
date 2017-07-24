import ctypes
from ycquant.yc_libs import *


class StrategyInfo(ctypes.Structure):
    _fields_ = [("profit_arr", ctypes.POINTER(ctypes.c_double)),
                ("op_arr", ctypes.POINTER(ctypes.c_double))]


class YCBenchmark:
    def __init__(self, strategy=CrossBarStrategy):
        self.strategy = strategy

    @staticmethod
    def to_int_arr(data, len):
        return np.array([data[i] for i in range(len)], dtype=int)

    @staticmethod
    def to_double_arr(data, len):
        return np.array([data[i] for i in range(len)], dtype=float)

    def bar_evaluate(self, price_table, y_pred_arr=None):

        if y_pred_arr is None:
            y_pred_arr = -1 * (np.delete(price_table, [len(price_table) - 1]) - np.delete(price_table, [0]))
            y_pred_arr[y_pred_arr == 0] = 1

        n_data = len(price_table)
        indices = np.arange(n_data)
        indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        price_table_ptr = price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        res = self.strategy.get_info(indices_pointer, y_pred_arr_pointer, n_data, price_table_ptr, 0, -1)

        return {
            "profit_arr": self.to_double_arr(res.contents.profit_arr, n_data),
            "op_arr": self.to_int_arr(res.contents.op_arr, n_data)}


class YCPerfect:
    _yc_perfect = True

    def predict(self, price_table):
        y_pred_arr = -1 * (np.delete(price_table, [len(price_table) - 1]) - np.delete(price_table, [0]))
        y_pred_arr[y_pred_arr == 0] = 1
        return y_pred_arr
