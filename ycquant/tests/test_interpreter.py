from ycquant.yc_common import *
from ycquant.yc_interpreter import *

PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
FITNESS_FUNC_KEY = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CHEAT_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEAHPEANH1HH@Z"
DATA_PATH = "./data/demo.csv"

MODEL_NAME = "1500520500"


def test_interpreter():
    prog = YCGP.load("outputs/exp_{suf}".format(suf=MODEL_NAME))

    formula = prog.__str__()
    mc_formula = YCInterpreter.mc_interprete(formula)

    with open("outputs/exp_{suf}/mc_formula.txt".format(suf=MODEL_NAME), "w") as f:
        f.write(mc_formula + ";")


test_interpreter()
