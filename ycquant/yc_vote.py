import numpy as np


class YCVote:
    @staticmethod
    def uniform_vote(all_op_list):
        return np.mean(all_op_list, axis=0)

    def __init__(self, est_list, get_op_func=None):
        self.est_list = est_list
        self.n_est = len(self.est_list)
        if self.n_est % 2 == 0:
            print("If there's a equilibrium, we will take it as 0.")

        self.get_op_func = get_op_func

    def predict_by_op(self, X):
        if self.get_op_func is None:
            print("Got to have a get_op_func!")
            exit()

    def predict_by_y(self, X):

        y_list = []

        for est in self.est_list:
            y_list.append(est.execute(X))

        return self.uniform_vote(y_list)
