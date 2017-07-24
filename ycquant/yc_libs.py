from ctypes import *

from ycquant.yc_io import *

YC_PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
YC_FUNC_KEY_GET_INFO = "?get_info@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_INFO_BY_OP = "?get_info_by_op@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_REWARD = "?get_reward@BarStrategy@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_OP_ARR = "?get_op_arr@BarStrategy@BackTesting@GPQuant@@SAPEANPEANH@Z"


def fillprototype(f, restype, argtypes):
    """
    """
    f.restype = restype
    f.argtypes = argtypes


libyc = cdll.LoadLibrary(YC_PATH_TO_DLL)


class StrategyInfo(Structure):
    _fields_ = [("profit_arr", POINTER(c_double)),
                ("op_arr", POINTER(c_double))]


class CrossBarStrategy:
    def __init__(self):
        self.__createform = 'python'

    @staticmethod
    def get_reward(*args):
        f = getattr(libyc, YC_FUNC_KEY_GET_REWARD)
        return f(*args)

    @staticmethod
    def get_info(*args):
        f = getattr(libyc, YC_FUNC_KEY_GET_INFO)
        return f(*args)

    @staticmethod
    def get_info_by_op(*args):
        f = getattr(libyc, YC_FUNC_KEY_GET_INFO_BY_OP)
        return f(*args)

    @staticmethod
    def get_op_arr(*args):
        f = getattr(libyc, YC_FUNC_KEY_GET_OP_ARR)
        return f(*args)


fillprototype(getattr(libyc, YC_FUNC_KEY_GET_INFO), POINTER(StrategyInfo),
              [POINTER(c_int), POINTER(c_double), c_int, POINTER(c_double), c_int, c_int])
fillprototype(getattr(libyc, YC_FUNC_KEY_GET_INFO_BY_OP), POINTER(StrategyInfo),
              [POINTER(c_int), POINTER(c_double), c_int, POINTER(c_double), c_int, c_int])
fillprototype(getattr(libyc, YC_FUNC_KEY_GET_REWARD), c_double,
              [POINTER(c_int), POINTER(c_double), c_int, POINTER(c_double), c_int, c_int])
fillprototype(getattr(libyc, YC_FUNC_KEY_GET_OP_ARR), POINTER(c_double),
              [POINTER(c_double), c_int])
