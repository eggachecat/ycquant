from ycquant.yc_fitness import *
from ycquant.yc_gp import *
from ycquant.yc_io import *

import time

ts = int(time.time())

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"


def test_best_result(y, y_pred):
    bool_sample_weight = np.array(np.ones(len(x_data), dtype=int), dtype=bool)
    indices = y[bool_sample_weight]
    y_pred_arr = y_pred[bool_sample_weight]

    indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    price_table_ptr = price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    res = metric.evluate(indices_pointer, y_pred_arr_pointer, len(indices), price_table_ptr, 0, 0)
    print(res)


x_data, price_table = read_unsupervised_data(DATA_PATH)
metric = YCFitness(FITNESS_FUNC_KEY, path_to_lib=PATH_TO_DLL)
gp = YCGP(price_table, metric)
gp.set_params(population_size=1000, generations=20, stopping_criteria=20000, parsimony_coefficient=100)
gp.fit(x_data)

y = np.arange(x_data.shape[0])

y_pred = gp.predict(x_data)
test_best_result(y, y_pred)

gp.save("outputs/exp_{suf}".format(suf=ts))
prog = gp.load("outputs/exp_{suf}".format(suf=ts))
print("-------------------")
y_pred = prog.execute(x_data)
test_best_result(y, y_pred)
