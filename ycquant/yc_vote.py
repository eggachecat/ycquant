from ycquant.yc_libs import *
from ctypes import *


class YCVote:
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
        if self.strategy is None:
            print("Got to have a get_op_func!")
            exit()

        op_arr_list = []
        for est in self.est_list:
            y_pred = est.execute(X)
            y_pred_pointer = y_pred.ctypes.data_as(POINTER(c_double))
            op_arr = self.strategy.get_op_arr(y_pred_pointer, len(y_pred))
            op_arr_list.append(op_arr)

        return self.uniform_vote(op_arr_list)

    def predict_by_y(self, X):

        y_list = []

        for est in self.est_list:
            y_list.append(est.execute(X))

        return self.uniform_vote(y_list)
