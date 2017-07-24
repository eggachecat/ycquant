from ycquant.yc_gp import *
from ycquant.yc_libs import *
import time

DATA_PATH = "./data/demo_data"

x_data, price_table = read_unsupervised_data(DATA_PATH)
metric = CrossBarStrategy

panish_list = [1, 0.5, 0.1, 0.01, 0.001, 0.0001]
for _ in range(len(panish_list)):
    ts = int(time.time())

    gp = YCGP(price_table, metric)
    gp.set_params(population_size=5000, generations=5, stopping_criteria=2000, parsimony_coefficient=panish_list[_], max_samples=1.0, verbose=0)
    gp.fit(x_data)

    gp.save("outputs/exp_{suf}".format(suf=ts))
    print(ts)


# ts = int(time.time())
# DATA_PATH = "./data/demo_data"
#
# x_data, price_table = read_unsupervised_data(DATA_PATH)
# metric = CrossBarStrategy


# def test_best_result(y, y_pred):
#     bool_sample_weight = np.array(np.ones(len(x_data), dtype=int), dtype=bool)
#     indices = y[bool_sample_weight]
#     y_pred_arr = y_pred[bool_sample_weight]
#
#     indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
#     y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     price_table_ptr = price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     res = metric.get_reward(indices_pointer, y_pred_arr_pointer, len(indices), price_table_ptr, 0, 0)
#
#
# x_data, price_table = read_unsupervised_data(DATA_PATH)
# metric = CrossBarStrategy
# gp = YCGP(price_table, metric)
# gp.set_params(population_size=5000, generations=20, stopping_criteria=2000, parsimony_coefficient=0.1, max_samples=1.0)
# gp.fit(x_data)
#
# y = np.arange(x_data.shape[0])
#
# y_pred = gp.predict(x_data)
# test_best_result(y, y_pred)
#
# print("experiment:　{ts}".format(ts=ts))
#
# gp.save("outputs/exp_{suf}".format(suf=ts))
# prog = gp.load("outputs/exp_{suf}".format(suf=ts))
# y_pred = prog.execute(x_data)
# test_best_result(y, y_pred)
