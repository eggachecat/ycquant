from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *
from ycquant.yc_io import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CHEAT_FUNC_KEY = "?get_info@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

MODEL_NAME = "1500607339"


def test_bar(y_pred):
    x_data, price_table = read_unsupervised_data(DATA_PATH)

    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

    benchmark_info = bm.bar_evaluate(price_table)
    benchmark_cum_arr = np.cumsum(benchmark_info["profit_arr"])

    predict_info = bm.bar_evaluate(price_table, y_pred)
    predict_cum_arr = np.cumsum(predict_info["profit_arr"])

    canvas = YCCanvas(shape=(3, 2))

    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), benchmark_info["profit_arr"], color="black", label="benchmark", sub_canvas_id=1)
    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), predict_info["profit_arr"], color="red", label="algorithm", sub_canvas_id=1)

    canvas.set_y_label("Profit", sub_canvas_id=1)
    canvas.set_title("In-sample Summary", sub_canvas_id=1)

    canvas.draw_square_function(benchmark_info["op_arr"], color="black", label="benchmark", sub_canvas_id=3)
    canvas.draw_square_function(predict_info["op_arr"], color="red", label="algorithm", sub_canvas_id=3)

    canvas.set_y_label("Decision", sub_canvas_id=3)

    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), benchmark_cum_arr, color="black", label="benchmark",
                              line_style="solid", sub_canvas_id=5)
    canvas.draw_line_chart_2d(range(1, 1 + len(price_table)), predict_cum_arr, color="red", label="algorithm",
                              sub_canvas_id=5)

    canvas.set_x_label("Time", sub_canvas_id=5)
    canvas.set_y_label("Accumlated Profit", sub_canvas_id=5)
    canvas.set_legend(sub_canvas_id=5)

    canvas.froze()


_x_data, _price_table = read_unsupervised_data(DATA_PATH)
prog = YCGP.load("outputs/exp_{suf}".format(suf=MODEL_NAME))
y_pred = prog.execute(_x_data)

test_bar(y_pred)
