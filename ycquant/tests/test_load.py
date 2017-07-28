from ycquant.yc_io import *


model_dir = "D:/sunao/workspace/python/ycquant/outputs/exp_"
exp_name = "1501211668"
model_name = model_dir + exp_name + "/model.pkl"

model = load_model(model_name)