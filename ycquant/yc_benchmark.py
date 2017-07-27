from ycquant.yc_backtest import *
import numpy as np


class YCPerfect:
    _yc_perfect = True

    def predict(self, price_table):
        future_price_table = np.roll(price_table, -1)
        y_pred = future_price_table - price_table
        y_pred[-1] = 0
        y_pred[y_pred == 0] = 1

        return y_pred


class YCFitPerfect:
    _yc_perfect = True

    def predict(self, y_true):
        return y_true
