from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *


def compare_performance(bm, y_pred, price_table, save_file_path=None):
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

    if save_file_path is None:
        pass
    else:
        canvas.save(save_file_path)

    canvas.froze()
