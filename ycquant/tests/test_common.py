from ycquant.yc_io import *
from ycquant.yc_common import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

CONST_MODEL_NAME = "1500535267"


def test_compare_performance_with_benchmark():
    x_data, price_table = read_unsupervised_data(DATA_PATH)

    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

    prog = YCGP.load("outputs/exp_{suf}".format(suf=CONST_MODEL_NAME))
    y_pred = prog.execute(x_data)

    compare_performance_with_benchmark(bm, y_pred, price_table, "outputs/exp_{suf}/performance.png".format(suf=CONST_MODEL_NAME))


def test_compare_performance_with_strategies():
    x_data, price_table = read_unsupervised_data(DATA_PATH)

    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

    profit_matrix = []
    model_name_list = ["1500535267", "1500520500"]
    for model_name in model_name_list:
        prog = YCGP.load("outputs/exp_{suf}".format(suf=model_name))
        y_pred = prog.execute(x_data)
        profit_matrix.append(bm.bar_evaluate(price_table, y_pred))

    compare_performance(profit_matrix)


test_compare_performance_with_benchmark()
