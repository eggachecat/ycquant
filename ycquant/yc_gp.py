import ctypes
import _pickle as cPickle

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

import numpy as np
import pydotplus
import os


class YCGP:
    """
    Gentic Programming Algorithm based on gplearn,
    which is able to handle changing target during the training (like reinforcement learning)
    """

    def __init__(self, price_table, metric, n_dim=0):
        """

        :param n_dim: int, default 0
            dimension of x_data
            currently is useless
        :param price_table: list of float
            the closed price corresponding to the x_data
            will be used to evaluate the fitenss

        """
        self.n_dim = n_dim
        self.price_table = price_table
        self.price_table_ptr = self.price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.len_price_table = len(price_table)
        self.metric = metric

        self.explict_fiteness_func = self.make_explict_func()
        self.est_gp = None

    def make_explict_func(self):
        n_dim = self.n_dim
        price_table_ptr = self.price_table_ptr
        reward_func = self.metric.evluate_func

        def explicit_fitness(y, y_pred, sample_weight):
            """

            :param y: as indicies correspondint to _y_pred
                    see fit() below
                e.g.
                    y = [2,5,7] and y_pred = [1.23, 2.34, 8.12]
                    means:
                        f(x[2]) = 1.23
                        f(x[5]) = 2.34
                        f(x[7]) = 8.12
            :param y_pred:
            :param sample_weight:
            :return:
            """
            y_pred[y_pred == 0] = 1

            bool_sample_weight = np.array(sample_weight, dtype=bool)
            indices = y[bool_sample_weight]
            y_pred_arr = y_pred[bool_sample_weight]

            indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            result = reward_func(indices_pointer, y_pred_arr_pointer, len(indices), price_table_ptr, n_dim, 0)

            return result

        return explicit_fitness

    def set_params(self, population_size=500, generations=10, stopping_criteria=10, p_crossover=0.7, p_subtree_mutation=0.1,
                   p_hoist_mutation=0.05, p_point_mutation=0.1, verbose=1, parsimony_coefficient=0.01, function_set=None):

        self.population_size = population_size
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.verbose = verbose
        self.parsimony_coefficient = parsimony_coefficient

        if function_set is None:
            function_set = ['add', 'sub', 'mul', 'div', 'sin']

        self.function_set = function_set

    def fit(self, x_data):

        if not hasattr(self, 'function_set'):
            print("Automatically initilizing....")
            self.set_params()

        metric = make_fitness(self.explict_fiteness_func, True)

        self.est_gp = SymbolicRegressor(population_size=self.population_size,
                                        generations=self.generations, stopping_criteria=self.stopping_criteria,
                                        p_crossover=self.p_crossover, p_subtree_mutation=self.p_subtree_mutation,
                                        p_hoist_mutation=self.p_hoist_mutation, p_point_mutation=self.p_point_mutation,
                                        metric=metric,
                                        function_set=self.function_set,
                                        verbose=self.verbose, parsimony_coefficient=self.parsimony_coefficient)

        self.est_gp.fit(x_data, np.arange(x_data.shape[0]))

        return self.est_gp

    def predict(self, x_data):

        return self.est_gp.predict(x_data)

    def save(self, folder_path):
        """Due to pickle not supporting saving ctype,
              so we have to save the best program  without metric function rather than whole regressor

          :param folder_path: str
                where to save the best est
          """

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_path = "{fd}/model.pkl".format(fd=folder_path)
        plot_path = "{fd}/expression.png".format(fd=folder_path)

        best_program = self.est_gp._program
        graph = pydotplus.graphviz.graph_from_dot_data(best_program.export_graphviz())
        graph.write_png(plot_path)

        best_program.metric = None
        with open(model_path, "wb") as f:
            cPickle.dump(best_program, f)

    @staticmethod
    def load(folder_path):
        """

        :param folder_path:  str
                where to save the best est
        :return: program
            program.execute(x_data) to get the predict
        """

        if not os.path.exists(folder_path):
            print("Folder: {fd} not exists!!".format(fd=folder_path))
        model_path = "{fd}/model.pkl".format(fd=folder_path)

        with open(model_path, "rb") as f:
            model = cPickle.load(f)
        return model
