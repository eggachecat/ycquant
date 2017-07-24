from ycquant.yc_io import *


DATA_PATH = "./data/demo.csv"

CONST_MODEL_NAME = "1500535267"

x_data, label = read_classification_data(DATA_PATH)
print(label)

