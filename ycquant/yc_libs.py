import ctypes

from ycquant.yc_io import *

YC_PATH_TO_DLL = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
YC_FUNC_KEY_GET_INFO = "?get_info@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_INFO_BY_OP = "?get_info_by_op@BarStrategy@BackTesting@GPQuant@@SAPEAUstrategy_info@3@PEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_REWARD = "?get_reward@BarStrategy@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
YC_FUNC_KEY_GET_OP_ARR = "?get_op_arr@BarStrategy@BackTesting@GPQuant@@SAPEANPEANH@Z"


