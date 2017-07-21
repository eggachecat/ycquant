from ycquant.yc_io import *
from ycquant.yc_common import *
from ycquant.yc_vote import *
PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BarStrategy@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?get_info@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
DECIDE_FUNC_KEY = ""
TRAIN_DATA_PATH = "./data/demo_data.train"
TEST_DATA_PATH = "./data/demo_data.test"

CONST_MODEL_NAME = "1500606856"


def test_compare_performance_with_benchmark(sample_size=1.0):
    y_pred_list = []
    price_table_list = []

    prog = YCGP.load("outputs/exp_{suf}".format(suf=CONST_MODEL_NAME))
    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)

    for file_name in [TRAIN_DATA_PATH, TEST_DATA_PATH]:
        x_data, price_table = read_unsupervised_data(file_name)

        y_pred = prog.execute(x_data)

        n_samples = int(sample_size * len(y_pred))
        indicies = sorted(np.random.choice(len(y_pred), n_samples, replace=False))

        y_pred = y_pred[indicies]
        price_table = price_table[indicies]

        y_pred_list.append(y_pred)
        price_table_list.append(price_table)

    compare_performance_with_benchmark(bm, y_pred_list, price_table_list, dataset_label_list=["In Sample", "Out of Sample"],
                                       save_file_path="outputs/exp_{suf}/performance.png".format(suf=CONST_MODEL_NAME))


def test_compare_performance_with_strategies(sample_size):
    performance_metric_matrix = []
    bm = YCBenchmark(CHEAT_FUNC_KEY, path_to_lib=PATH_TO_DLL)
    model_name_list = ["Doctor Strange", "1500625233", "1500625397", "1500625414", "1500625432"]
    file_name_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]

    if not type(sample_size) is list:
        sample_size_list = [sample_size for _ in len(file_name_list)]
    else:
        sample_size_list = sample_size

    for i in range(len(file_name_list)):

        file_name = file_name_list[i]
        p_sample_size = sample_size_list[i]

        x_data, price_table = read_unsupervised_data(file_name)
        performance_metric_list = [bm.bar_evaluate(price_table)]

        n_samples = int(p_sample_size * len(price_table))
        print(n_samples)

        indicies = sorted(np.random.choice(len(price_table), n_samples, replace=False))

        for model_name in model_name_list[1:]:
            prog = YCGP.load("outputs/exp_{suf}".format(suf=model_name))
            y_pred = prog.execute(x_data)

            performance_metric_list.append(bm.bar_evaluate(price_table[indicies], y_pred[indicies]))

        performance_metric_list.append(uniform_vote(performance_metric_list[1:]))

        performance_metric_matrix.append(performance_metric_list)

    model_name_list.append("voting")
    compare_performance(performance_metric_matrix, label_list=model_name_list, dataset_label_list=["In Sample", "Out of Sample"])


test_compare_performance_with_strategies([0.0001, 1])
