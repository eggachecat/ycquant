from ycquant.yc_io import *
from ycquant.yc_common import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

MODEL_NAME = "1500520500"

x_data, price_table = read_unsupervised_data(DATA_PATH)

bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

prog = YCGP.load("outputs/exp_{suf}".format(suf=MODEL_NAME))
y_pred = prog.execute(x_data)

compare_performance(bm, y_pred, price_table, "outputs/exp_{suf}/performance.png".format(suf=MODEL_NAME))
