from ycquant.yc_libs import *
from ctypes import *


class YCVote:
    _yc_vote = True

    @staticmethod
    def uniform_vote(all_op_list):
        return np.mean(all_op_list, axis=0)

    def __init__(self, est_list, strategy=CrossBarStrategy):
        self.est_list = est_list
        self.n_est = len(self.est_list)
        if self.n_est % 2 == 0:
            print("If there's a equilibrium, we will take it as 0.")
        self.strategy = strategy

    def predict_by_op(self, X):
        op_arr_list = []
        for est in self.est_list:
            if hasattr(est, "_yc_gp"):
                y_pred = est.predict(X)
                print("_yc_gp", len(y_pred))
            else:
                # y_as_stock_price
                stock_price_pred = est.predict(X)
                y_pred = -1 * (np.delete(stock_price_pred, [len(stock_price_pred) - 1]) - np.delete(stock_price_pred, [0]))
                y_pred[y_pred == 0] = 1
                y_pred = np.insert(y_pred, [0], [y_pred[0]])
                print("_yc_gp", len(y_pred))

            res = self.strategy.get_op_arr(y_pred, len(y_pred))
            op_arr_list.append(res)
        democracy = self.uniform_vote(op_arr_list)
        print(democracy)
        return self.strategy.get_info_by_op()

    def predict(self, X):

        y_list = []

        for est in self.est_list:
            if hasattr(est, "_yc_gp"):
                y_pred = est.predict(X)
                print("_yc_gp", len(y_pred))

            else:
                # y_as_stock_price
                stock_price_pred = est.predict(X)
                y_pred = -1 * (np.delete(stock_price_pred, [len(stock_price_pred) - 1]) - np.delete(stock_price_pred, [0]))
                y_pred[y_pred == 0] = 1
                y_pred = np.insert(y_pred, [0], [y_pred[0]])
                print("_yc_gp", len(y_pred))
            y_list.append(y_pred)

        return self.uniform_vote(y_list)
