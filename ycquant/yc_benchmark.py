from ycquant.yc_backtest import *


class YCPerfect:
    _yc_perfect = True

    def predict(self, price_table):
        y_pred_arr = -1 * (np.delete(price_table, [len(price_table) - 1]) - np.delete(price_table, [0]))
        y_pred_arr[y_pred_arr == 0] = 1
        return y_pred_arr
