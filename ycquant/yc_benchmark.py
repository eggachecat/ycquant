from ycquant.yc_fitness import *
import numpy as np


class YCBenchmark:
    def __init__(self, func_key, func_config=None, path_to_lib="libs/GPQuant.dll"):
        """

                :param func_key: str
                        attr of the metric function in dll
                :param func_config: dict, default None
                    config the metric function
                :param path_to_lib: str, default path_to_libs
                    where is the dll
                """
        self.lib = ctypes.cdll.LoadLibrary(path_to_lib)
        self.cheat_func = getattr(self.lib, func_key)

        if func_config is None:
            func_config = {
                "argtypes": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int],
                "restype": ctypes.POINTER(ctypes.c_double)
            }

        for attr in func_config:
            setattr(self.cheat_func, attr, func_config[attr])

    def bar_evaluate(self, price_table, y_pred_arr=None):

        if y_pred_arr is None:
            y_pred_arr = -1 * (price_table - price_table[0])
        n_data = len(price_table)

        indices = np.arange(n_data)
        indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        price_table_ptr = price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        res = self.cheat_func(indices_pointer, y_pred_arr_pointer, n_data, price_table_ptr, 0, 1)

        return [float(res[i]) for i in range(n_data)]
