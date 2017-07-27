from ycquant.yc_performance import *
from ycquant.yc_interpreter import *


DATA_PATH = "./data/demo.csv"

MODEL_NAME = "1500520500"


def test_interpreter():
    prog = YCGP.load("outputs/exp_{suf}".format(suf=MODEL_NAME))

    formula = prog.__str__()
    mc_formula = YCInterpreter.mc_interprete(formula)

    with open("outputs/exp_{suf}/mc_formula.txt".format(suf=MODEL_NAME), "w") as f:
        f.write("f[1]=" + mc_formula + ";")


test_interpreter()
