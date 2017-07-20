from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *


def compare_performance(profit_matrix, label_list=None, save_file_path=None):
    canvas = YCCanvas(shape=(2, 1))

    n_strategies = len(profit_matrix)
    n_dates = len(profit_matrix[0])

    if label_list is None:
        label_list = ["strategy-{n}".format(n=i) for i in range(n_strategies)]

    if not len(label_list) == n_strategies:
        print("length of labels doesnot match that of y_pred_matrix")
        exit()

    for i in range(n_strategies):
        predict_profit_arr = profit_matrix[i]
        predict_cum_arr = np.cumsum(predict_profit_arr)

        canvas.draw_line_chart_2d(range(1, 1 + n_dates), predict_profit_arr, label=label_list[i],
                                  sub_canvas_id=1)

        canvas.draw_line_chart_2d(range(1, 1 + n_dates), predict_cum_arr, label=label_list[i],
                                  sub_canvas_id=2)

    canvas.set_x_label("Indices", sub_canvas_id=1)
    canvas.set_y_label("Profit", sub_canvas_id=1)
    canvas.set_legend(sub_canvas_id=1)

    canvas.set_x_label("Indices", sub_canvas_id=2)
    canvas.set_y_label("Accumlated Profit", sub_canvas_id=2)
    canvas.set_legend(sub_canvas_id=2)

    if save_file_path is None:
        pass
    else:
        canvas.save(save_file_path)

    canvas.froze()


def compare_performance_with_benchmark(bm, y_pred, price_table, save_file_path=None):
    benchmark_profit_arr = bm.bar_evaluate(price_table)
    predict_profit_arr = bm.bar_evaluate(price_table, y_pred)
    profit_matrix = [benchmark_profit_arr, predict_profit_arr]

    compare_performance(profit_matrix, label_list=["benchmark", "strategy"], save_file_path=save_file_path)
