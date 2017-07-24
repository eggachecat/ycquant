from ycquant.yc_tree import *
from ycquant.yc_io import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

CONST_MODEL_NAME = "1500535267"

x_data, label = read_classification_data(DATA_PATH)
yc = YCRandomForest()