from ycquant.yc_splitter import *
from ycquant.yc_common import *
import time
from ycquant.yc_vote import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

ts = int(time.time())

train_file_name_1, test_file_name_1 = split_train_and_test("data/product_01", header=None, sep="  ", split_ratio=1.0)
train_file_name_2, test_file_name_2 = split_train_and_test("data/product_02", header=None, sep="  ", split_ratio=1.0)

x_data_1, price_table_1 = read_unsupervised_data(train_file_name_1)
x_data_2, price_table_2 = read_unsupervised_data(train_file_name_2)
print(price_table_1)
print(price_table_2)

price_table_all = np.hstack((price_table_1, price_table_2))
print(price_table_all)

x_data_all = np.vstack((x_data_1, x_data_2))

metric = CrossBarStrategy

gp_1 = YCGP(price_table_1, metric)
gp_1.set_params(population_size=3000, generations=20, stopping_criteria=2000000, parsimony_coefficient=300, max_samples=1.0)
gp_1.fit(x_data_1)
gp_1.save("outputs/exp_{suf}_1".format(suf=ts))

gp_2 = YCGP(price_table_2, metric)
gp_2.set_params(population_size=3000, generations=20, stopping_criteria=2000000, parsimony_coefficient=3000, max_samples=1.0)
gp_2.fit(x_data_2)
gp_2.save("outputs/exp_{suf}_2".format(suf=ts))

split_list = [0, len(price_table_1), len(price_table_all)]
reward_weight_list = [0.5, 0.5]
gp_all = YCGP(price_table_all, metric, split_list=split_list, reward_weight=reward_weight_list)
gp_all.set_params(population_size=2000, generations=20, stopping_criteria=2000000, parsimony_coefficient=1000, max_samples=1.0)
gp_all.fit(x_data_all)

gp_all.save("outputs/exp_{suf}_all".format(suf=ts))

x_reg_1, y_reg_1 = read_regression_data(train_file_name_1)
x_reg_2, y_reg_2 = read_regression_data(train_file_name_2)

x_reg_all = np.vstack((x_reg_1, x_reg_2))
y_reg_all = np.hstack((y_reg_1, y_reg_2))

lr_1 = LinearRegression()
lr_1.fit(x_reg_1, y_reg_1)

lr_2 = LinearRegression()
lr_2.fit(x_reg_2, y_reg_2)

lr_all = LinearRegression()
lr_all.fit(x_reg_all, y_reg_all)

gbr_1 = GradientBoostingRegressor()
gbr_1.fit(x_reg_1, y_reg_1)

gbr_2 = GradientBoostingRegressor()
gbr_2.fit(x_reg_2, y_reg_2)

gbr_all = GradientBoostingRegressor()
gbr_all.fit(x_reg_all, y_reg_all)

voter_1 = YCVote([gp_1, lr_1, gbr_1])
voter_2 = YCVote([gp_2, lr_2, gbr_2])
voter_all = YCVote([gp_all, lr_all, gbr_all])

perfect = YCPerfect()

train_file_name_3, test_file_name_3 = split_train_and_test("data/product_03.csv", header=None, sep=",", split_ratio=1.0)

compare_performance_([gp_1, gp_2, lr_1, lr_2, gbr_1, gbr_2, lr_all, gbr_all, gp_all, voter_1, voter_2, voter_all],
                     [train_file_name_3],
                     ["In Sample of data-3"],
                     ["gp_1", "gp_2", "lr_1", "lr_2", "gbr_1", "gbr_2", "lr_all", "gbr_all", "gp_all", "v_1", "v_2", "v_all"],
                     use_price_table_list=False,
                     save_file_path="outputs/multi.png")
#
# compare_performance_([perfect, gp_1, gp_2, lr_1, lr_2, gbr_1, gbr_2, lr_all, gbr_all, gp_all, voter_1, voter_2, voter_all],
#                      [train_file_name_3],
#                      ["In Sample of data-1", "In Sample of data-2", "In Sample of data-3"],
#                      ["Perfect", "gp_1", "gp_2", "lr_1", "lr_2", "gbr_1", "gbr_2", "lr_all", "gbr_all", "gp_all", "v_1", "v_2", "v_all"],
#                      use_price_table_list=False,
#                      save_file_path="outputs/multi.png")

#
# x_reg_1, y_reg_1 = read_regression_data(train_file_name_1)
# x_reg_2, y_reg_2 = read_regression_data(train_file_name_2)
#
# x_reg_all = np.vstack((x_reg_1, x_reg_2))
# y_reg_all = np.hstack((y_reg_1, y_reg_2))
#
# lr_1 = LinearRegression()
# lr_1.fit(x_reg_1, y_reg_1)
#
# lr_2 = LinearRegression()
# lr_2.fit(x_reg_2, y_reg_2)
#
# lr_all = LinearRegression()
# lr_all.fit(x_reg_all, y_reg_all)
#
# gbr_1 = GradientBoostingRegressor()
# gbr_1.fit(x_reg_1, y_reg_1)
#
# gbr_2 = GradientBoostingRegressor()
# gbr_2.fit(x_reg_2, y_reg_2)
#
# gbr_all = GradientBoostingRegressor()
# gbr_all.fit(x_reg_all, y_reg_all)
#
# perfect = YCPerfect()
# compare_performance_([perfect, gp, lr_1, lr_2, lr_all, gbr_1, gbr_2, gbr_all],
#                      [train_file_name_1, test_file_name_1, train_file_name_2, test_file_name_2],
#                      ["In Sample of data-1", "Out of Sample of data-1", "In Sample of data-2", "Out of Sample of data-2"],
#                      ["Perfect", "Genetic Programming", "LR-1", "LR-2", "LR-all", "GBR-1", "GBR-2", "GBR-all"], use_price_table_list=False,
#                      save_file_path="outputs/multi.png")
