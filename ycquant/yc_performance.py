from ycquant.yc_plot import *
from ycquant.yc_gp import *
from ycquant.yc_io import *


def compare_performance(est_list, file_name_list, dataset_label_list, model_name_list, strategy=CrossBarStrategy, use_price_table_list=True,
                        save_file_path=None):
    performance_metric_matrix = []

    price_table_list = []

    for i in range(len(file_name_list)):

        file_name = file_name_list[i]

        x_data, price_table = read_unsupervised_data(file_name)
        price_table_list.append(price_table)
        performance_metric_list = []

        for est in est_list:
            if hasattr(est, "_yc_perfect"):
                y_pred = est.predict(price_table)
            else:
                y_pred = est.predict(x_data)
            print(type(est), len(y_pred), len(price_table))
            performance_metric_list.append(strategy.get_info(y_pred, price_table))
        performance_metric_matrix.append(performance_metric_list)

    if not use_price_table_list:
        price_table_list = None

    compare_performance_plot(performance_metric_matrix, label_list=model_name_list, dataset_label_list=dataset_label_list,
                             save_file_path=save_file_path,
                             price_table_list=price_table_list)


def compare_performance_plot(performance_metric_matrix, label_list=None, dataset_label_list=None, save_file_path=None, price_table_list=None):
    n_dataset = len(performance_metric_matrix)
    canvas = YCCanvas(shape=(3, n_dataset))

    for _ in range(n_dataset):

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
            print(profit_arr)
            cum_arr = np.cumsum(profit_arr)
            print(cum_arr)

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

        canvas.set_title("In dataset-{name}".format(name=dataset_label_list[_]), _sub_canvas_id)

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
