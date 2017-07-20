from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *
from ycquant.yc_io import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

MODEL_NAME = "1500520500"


def test_bar(y_pred):
    x_data, price_table = read_unsupervised_data(DATA_PATH)

    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

    benchmark_profit_arr = bm.bar_evaluate(price_table)
    benchmark_cum_arr = np.cumsum(benchmark_profit_arr)

    predict_profit_arr = bm.bar_evaluate(price_table, y_pred)
    predict_cum_arr = np.cumsum(predict_profit_arr)

    canvas = YCCanvas(shape=(2, 1))

    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), benchmark_profit_arr, color="blue", label="benchmark",
                              line_style="solid", sub_canvas_id=1)
    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), predict_profit_arr, color="red", label="algorithm",
                              sub_canvas_id=1)

    canvas.set_x_label("Indices", sub_canvas_id=1)
    canvas.set_y_label("Profit", sub_canvas_id=1)
    canvas.set_legend(sub_canvas_id=1)

    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), benchmark_cum_arr, color="blue", label="benchmark",
                              line_style="solid", sub_canvas_id=2)
    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), predict_cum_arr, color="red", label="algorithm",
                              sub_canvas_id=2)

    canvas.set_x_label("Indices", sub_canvas_id=2)
    canvas.set_y_label("Accumlated Profit", sub_canvas_id=2)
    canvas.set_legend(sub_canvas_id=2)

    canvas.froze()


_x_data, _price_table = read_unsupervised_data(DATA_PATH)
prog = YCGP.load("outputs/exp_{suf}".format(suf=MODEL_NAME))
y_pred = prog.execute(_x_data)

test_bar(y_pred)
