from ycquant.yc_splitter import *
from ycquant.yc_gp import *
from ycquant.yc_libs import *
from ycquant.yc_common import *
import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

ts = int(time.time())

train_file_name, test_file_name = split_train_and_test("data/product_02", header=None, sep="  ")

x_data, price_table = read_unsupervised_data(train_file_name)
metric = CrossBarStrategy
gp = YCGP(price_table, metric)
gp.set_params(population_size=5000, generations=20, stopping_criteria=2000000, parsimony_coefficient=50, max_samples=1.0)
gp.fit(x_data)
gp.save("outputs/exp_{suf}".format(suf=ts))

x_data_reg, y_reg = read_regression_data(train_file_name)

lr = LinearRegression()
lr.fit(x_data_reg, y_reg)
print("lr done")

gbr = GradientBoostingRegressor()
gbr.fit(x_data_reg, y_reg)

perfect = YCPerfect()
compare_performance_([perfect, gp, lr, gbr], [train_file_name, test_file_name], ["In Sample", "Out of Sample"],
                     ["Perfect", "Genetic Programming", "Linear Regression", "Gradient Boosting Regression"], use_price_table_list=False)



# x_data_reg, y_reg = read_regression_data(train_file_name)
# rf_est = RandomForestClassifier(n_estimators=3, max_features=None)
# rf_est.fit(x_data_reg, y_reg)
# print("rf done")
#
# lr = LinearRegression()
# lr.fit(x_data_reg, y_reg)
# print("lr done")
#
# gbr = GradientBoostingRegressor()
# gbr.fit(x_data_reg, y_reg)
# print("gbr done")
#
# hold = YCHold()
# perfect = YCPerfect()
# compare_performance_([hold, perfect, gp, rf_est, lr, gbr], [train_file_name, test_file_name], ["In Sample", "Out of Sample"],
#                      ["hold", "Perfect", "Genetic Programming", "Random Forest Regression", "Linear Regression", "Gradient Boosting Regression"])
