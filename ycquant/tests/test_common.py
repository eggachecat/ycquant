from ycquant.yc_io import *
from ycquant.yc_performance import *
from ycquant.yc_ensumble import *

TRAIN_DATA_PATH = "./data/product_01.train"
TEST_DATA_PATH = "./data/product_01.test"

CONST_MODEL_NAME = "1500876944"


def test_compare_performance_with_strategies(sample_size, model_name_list, voters):
    performance_metric_matrix = []
    strategy = CrossBarStrategy
    model_name_list = ["Doctor Strange", "1500864834", "1500864846", "1500864857", "1500864870", "1500864885"]

    voters = ["1500864834", "1500864846", "1500864857", "1500864870", "1500864885"]
    est_list = []
    for voter in voters:
        est_list.append(YCGP.load("outputs/exp_{suf}".format(suf=voter)))
    v = YCVote(est_list)

    file_name_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]

    if not type(sample_size) is list:
        sample_size_list = [sample_size for _ in len(file_name_list)]
    else:
        sample_size_list = sample_size

    for i in range(len(file_name_list)):

        file_name = file_name_list[i]
        p_sample_size = sample_size_list[i]

        x_data, price_table = read_unsupervised_data(file_name)
        performance_metric_list = [strategy.get_info(price_table)]

        n_samples = int(p_sample_size * len(price_table))
        print(n_samples)

        indicies = sorted(np.random.choice(len(price_table), n_samples, replace=False))

        for model_name in model_name_list[1:]:
            prog = YCGP.load("outputs/exp_{suf}".format(suf=model_name))
            y_pred = prog.predict(x_data)

            performance_metric_list.append(strategy.get_info(price_table[indicies], y_pred[indicies]))
        performance_metric_list.append(strategy.get_info(price_table[indicies], v.predict_by_y(x_data)[indicies]))
        performance_metric_matrix.append(performance_metric_list)

    model_name_list.append("votes")
    compare_performance_plot(performance_metric_matrix, label_list=model_name_list, dataset_label_list=["In Sample", "Out of Sample"])

#
# def test_compare_performance_within_generation(model_name, top, sample_size):
#     performance_metric_matrix = []
#     bm = YCEvaluation(CrossBarStrategy)
#     model_name_list = ["Doctor Strange", model_name]
#     est_list = YCGP.load_generation("outputs/exp_{suf}".format(suf=model_name))
#     v = YCVote(est_list[:top])
#
#     file_name_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
#
#     if not type(sample_size) is list:
#         sample_size_list = [sample_size for _ in len(file_name_list)]
#     else:
#         sample_size_list = sample_size
#
#     for i in range(len(file_name_list)):
#
#         file_name = file_name_list[i]
#         p_sample_size = sample_size_list[i]
#
#         x_data, price_table = read_unsupervised_data(file_name)
#         performance_metric_list = [bm.bar_evaluate(price_table)]
#
#         n_samples = int(p_sample_size * len(price_table))
#
#         indicies = sorted(np.random.choice(len(price_table), n_samples, replace=False))
#
#         for model_name in model_name_list[1:]:
#             prog = YCGP.load("outputs/exp_{suf}".format(suf=model_name))
#             y_pred = prog.execute(x_data)
#             performance_metric_list.append(bm.bar_evaluate(price_table[indicies], y_pred[indicies]))
#         performance_metric_list.append(bm.bar_evaluate(price_table[indicies], v.predict_by_y(x_data)[indicies]))
#         performance_metric_matrix.append(performance_metric_list)
#     model_name_list.append("generation voting")
#     compare_performance(performance_metric_matrix, label_list=model_name_list, dataset_label_list=["In Sample", "Out of Sample"])

# test_compare_performance_with_strategies([1, 1])
# test_compare_performance_within_generation("1500861636", top=10, sample_size=[1, 1])
