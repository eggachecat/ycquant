from ycquant.yc_gp import *
from ycquant.yc_libs import *
import time
from ctypes import *

DATA_PATH = "./data/product_01.train"

x_data, price_table = read_unsupervised_data(DATA_PATH, sep=',', header=None)
metric = CrossBarStrategy
print(x_data)
print("----------")
print(price_table, len(price_table))

ts = int(time.time())


def test_best_result(y, y_pred):
    bool_sample_weight = np.array(np.ones(len(x_data), dtype=int), dtype=bool)
    indices = y[bool_sample_weight]
    y_pred_arr = y_pred[bool_sample_weight]

    indices_pointer = indices.ctypes.data_as(POINTER(c_int))
    y_pred_arr_pointer = y_pred_arr.ctypes.data_as(POINTER(c_double))
    price_table_ptr = price_table.ctypes.data_as(POINTER(c_double))
    res = metric.get_reward(indices_pointer, y_pred_arr_pointer, len(indices), price_table_ptr, 0, 0)
    print(res)


x_data, price_table = read_unsupervised_data(DATA_PATH)
metric = CrossBarStrategy
gp = YCGP(price_table, metric)
gp.set_params(population_size=1000, generations=20, stopping_criteria=2000, parsimony_coefficient=0, max_samples=1.0)
gp.fit(x_data)

y = np.arange(x_data.shape[0])

y_pred = gp.predict(x_data)
test_best_result(y, y_pred)

print("experiment:　{ts}".format(ts=ts))

gp.save("outputs/exp_{suf}".format(suf=ts))
prog = gp.load("outputs/exp_{suf}".format(suf=ts))
y_pred = prog.execute(x_data)
test_best_result(y, y_pred)
