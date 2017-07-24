from ycquant.yc_benchmark import *
from ycquant.yc_plot import *
from ycquant.yc_gp import *


def compare_performance_(est_list, file_name_list, dataset_label_list, model_name_list, use_price_table_list=True):
    performance_metric_matrix = []
    bm = YCBenchmark(CrossBarStrategy)

    price_table_list = []

    for i in range(len(file_name_list)):

        file_name = file_name_list[i]

        x_data, price_table = read_unsupervised_data(file_name)
        price_table_list.append(price_table)
        performance_metric_list = []

        for est in est_list:

            if hasattr(est, "_yc_perfect"):
                y_pred = est.predict(price_table)
                performance_metric_list.append(bm.bar_evaluate(price_table, y_pred))
            elif hasattr(est, "_yc_gp"):
                y_pred = est.predict(x_data)
                performance_metric_list.append(bm.bar_evaluate(price_table, y_pred))
            else:
                # y_as_stock_price
                stock_price_pred = est.predict(x_data)
                y_pred_arr = -1 * (np.delete(stock_price_pred, [len(stock_price_pred) - 1]) - np.delete(stock_price_pred, [0]))
                y_pred_arr[y_pred_arr == 0] = 1
                performance_metric_list.append(bm.bar_evaluate(price_table, y_pred_arr))

        performance_metric_matrix.append(performance_metric_list)

    if not use_price_table_list:
        price_table_list = None

    compare_performance(performance_metric_matrix, label_list=model_name_list, dataset_label_list=dataset_label_list,
                        price_table_list=price_table_list)


def compare_performance(performance_metric_matrix, label_list=None, dataset_label_list=None, save_file_path=None, price_table_list=None):
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

        print(len(label_list), n_strategies)
        if not len(label_list) == n_strategies:
            print("length of labels doesnot match that of y_pred_matrix")
            exit()

        _sub_canvas_id = _ + 1

        for i in range(n_strategies):
            profit_arr = performance_metric_arr[i]["profit_arr"]
            cum_arr = np.cumsum(profit_arr)
            op_arr = performance_metric_arr[i]["op_arr"]

            print(len(profit_arr), "n_dates", n_dates)
            canvas.draw_line_chart_2d(range(1, 1 + n_dates), profit_arr, label=label_list[i],
                                      sub_canvas_id=_sub_canvas_id)
            canvas.draw_square_function(op_arr * float(1 + i / n_strategies), label=label_list[i], sub_canvas_id=_sub_canvas_id + n_dataset * 1)

            canvas.draw_line_chart_2d(range(1, 1 + n_dates), cum_arr, label=label_list[i],
                                      sub_canvas_id=_sub_canvas_id + n_dataset * 2)

        if price_table_list is not None:
            canvas.draw_line_chart_2d(range(1, 1 + n_dates), price_table_list[_], label="Buy and hold",
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
