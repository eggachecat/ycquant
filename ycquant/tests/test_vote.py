from ycquant.yc_ensumble import *
from ycquant.yc_gp import *
from ycquant.yc_io import *
voters = ["1500861514", "1500861608", "1500861636"]

TRAIN_DATA_PATH = "./data/demo_data.train"
TEST_DATA_PATH = "./data/demo_data.test"
x_data, price_table = read_unsupervised_data(TEST_DATA_PATH)

est_list = []
for voter in voters:
    est_list.append(YCGP.load("outputs/exp_{suf}".format(suf=voter)))
v = YCVote(est_list)
# print(v.predict_by_op(x_data))
print(v.predict_by_y(x_data))