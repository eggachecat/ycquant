from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *


def compare_performance(performance_metric_matrix, label_list=None, dataset_label_list=None, save_file_path=None):
    n_dataset = len(performance_metric_matrix)
    canvas = YCCanvas(shape=(3, n_dataset))

    for _ in range(n_dataset):

        # profit_matrix = performance_metric_matrix[_].profit_matrix
        # op_matrix = performance_metric_matrix[_].op_matrix

        performance_metric_arr = performance_metric_matrix[_]

        n_strategies = len(performance_metric_arr)
        n_dates = len(performance_metric_arr[0]["profit_arr"])

        if label_list is None:
            label_list = ["strategy-{n}".format(n=i) for i in range(n_strategies)]

        if not len(label_list) == n_strategies:
            print("length of labels doesnot match that of y_pred_matrix")
            exit()

        _sub_canvas_id = _ + 1

        for i in range(n_strategies):
            profit_arr = performance_metric_arr[i]["profit_arr"]
            cum_arr = np.cumsum(profit_arr)
            op_arr = performance_metric_arr[i]["op_arr"]

            canvas.draw_line_chart_2d(range(1, 1 + n_dates), profit_arr, label=label_list[i],
                                      sub_canvas_id=_sub_canvas_id)
            canvas.draw_square_function(op_arr * float(1 + i / n_strategies), label=label_list[i], sub_canvas_id=_sub_canvas_id + n_dataset * 1)

            canvas.draw_line_chart_2d(range(1, 1 + n_dates), cum_arr, label=label_list[i],
                                      sub_canvas_id=_sub_canvas_id + n_dataset * 2)

        canvas.set_title("Comparation of performance in dataset-{name}".format(name=dataset_label_list[_]), _sub_canvas_id)

        canvas.set_y_label("Profit", sub_canvas_id=_sub_canvas_id)
        canvas.set_y_label("Operation", sub_canvas_id=_sub_canvas_id + n_dataset * 1)
        canvas.set_y_label("Accumlated Profit", sub_canvas_id=_sub_canvas_id + n_dataset * 2)

        canvas.set_x_label("Indices", sub_canvas_id=_sub_canvas_id + n_dataset * 2)
        canvas.set_legend(sub_canvas_id=_sub_canvas_id + n_dataset * 2)

    if save_file_path is None:
        pass
    else:
        canvas.save(save_file_path)

    canvas.froze()


def compare_performance_with_benchmark(bm, y_pred_list, price_table_list, dataset_label_list, save_file_path=None):
    performance_metric_matrix = []

    if not type(price_table_list) is list:
        price_table_list = [price_table_list]

    n_dataset = len(price_table_list)

    for i in range(n_dataset):
        price_table = price_table_list[i]
        y_pred = y_pred_list[i]

        benchmark_metric = bm.bar_evaluate(price_table)
        predict_metric = bm.bar_evaluate(price_table, y_pred)

        performance_metric_matrix.append([benchmark_metric, predict_metric])

    compare_performance(performance_metric_matrix, label_list=["Doctor Strange ", "Our Strategy"], dataset_label_list=dataset_label_list,
                        save_file_path=save_file_path)

#
# def compare_performance(profit_matrix, label_list=None, save_file_path=None):
#     canvas = YCCanvas(shape=(2, 1))
#
#     n_strategies = len(profit_matrix)
#     n_dates = len(profit_matrix[0])
#
#     if label_list is None:
#         label_list = ["strategy-{n}".format(n=i) for i in range(n_strategies)]
#
#     if not len(label_list) == n_strategies:
#         print("length of labels doesnot match that of y_pred_matrix")
#         exit()
#
#     for i in range(n_strategies):
#         predict_profit_arr = profit_matrix[i]
#         predict_cum_arr = np.cumsum(predict_profit_arr)
#
#         canvas.draw_line_chart_2d(range(1, 1 + n_dates), predict_profit_arr, label=label_list[i],
#                                   sub_canvas_id=1)
#
#         canvas.draw_line_chart_2d(range(1, 1 + n_dates), predict_cum_arr, label=label_list[i],
#                                   sub_canvas_id=2)
#
#     canvas.set_x_label("Indices", sub_canvas_id=1)
#     canvas.set_y_label("Profit", sub_canvas_id=1)
#     canvas.set_legend(sub_canvas_id=1)
#
#     canvas.set_x_label("Indices", sub_canvas_id=2)
#     canvas.set_y_label("Accumlated Profit", sub_canvas_id=2)
#     canvas.set_legend(sub_canvas_id=2)
#
#     if save_file_path is None:
#         pass
#     else:
#         canvas.save(save_file_path)
#
#     canvas.froze()
#
#
# def compare_performance_with_benchmark(bm, y_pred, price_table, save_file_path=None):
#     benchmark_profit_arr = bm.bar_evaluate(price_table)
#
#     predict_profit_arr = bm.bar_evaluate(price_table, y_pred)
#     profit_matrix = [benchmark_profit_arr, predict_profit_arr]
#
#     compare_performance(profit_matrix, label_list=["benchmark", "strategy"], save_file_path=save_file_path)
