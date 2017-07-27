from ycquant.yc_preprocessing import *
from ycquant.yc_gp import *
from ycquant.yc_backtest import *
from ycquant.yc_io import *
from ycquant.yc_benchmark import *
from ycquant.yc_performance import *
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import os

script_dir = os.path.dirname(__file__)


class CrossBarStrategyShell:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)

    def predict(self, X):
        y_pred = self.est.predict(X)
        return CrossBarStrategy.price_to_movement(y_pred)


def main():
    ts = int(time.time())
    file_path = "data/product_01"

    abs_file_path = os.path.join(script_dir, file_path)

    train_file_name, test_file_name = split_train_and_test(abs_file_path, header=None, sep="  ")

    x_data, price_table = read_unsupervised_data(train_file_name)

    metric = CrossBarStrategy

    gp = YCGP(price_table, metric)
    gp.set_params(population_size=3000, generations=10, stopping_criteria=1000000, parsimony_coefficient=500, max_samples=1.0,
                  greater_is_better=True)
    gp.fit(x_data)
    gp.save("outputs/exp_{suf}".format(suf=ts))

    # oob_gp = gp.get_best_oob_est()
    # formula = oob_gp.__str__()
    # mc_formula = YCInterpreter.mc_interprete(formula)
    # print("oob", mc_formula)

    x_data_reg, y_reg = read_regression_data(train_file_name)

    lr = CrossBarStrategyShell(LinearRegression())
    lr.fit(x_data_reg, y_reg)

    gbr = CrossBarStrategyShell(GradientBoostingRegressor())
    gbr.fit(x_data_reg, y_reg)

    perfect = YCPerfect()

    model_list = [perfect, gp, lr, gbr]
    model_list_name = ["Perfect", "Genetic Programming", "Linear Regression", "Gradient Boosting Regression"]

    # model_list = [perfect, gp]
    # model_list_name = ["Perfect", "Genetic Programming"]

    compare_performance(model_list, [train_file_name, test_file_name], ["In Sample", "Out of Sample"], model_list_name, use_price_table_list=False)


if __name__ == "__main__":
    main()
