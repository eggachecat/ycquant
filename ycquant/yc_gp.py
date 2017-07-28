import _pickle as cPickle

from YCgplearn.genetic import SymbolicRegressor
from YCgplearn.fitness import make_fitness

import pydotplus

from ycquant.yc_interpreter import *
import os

# from ctypes import *
from ycquant.yc_backtest import *
import numpy as np


class YCGP:
    """
    Gentic Programming Algorithm based on gplearn,
    which is able to handle changing target during the training (like reinforcement learning)
    """
    _yc_gp = True

    def __init__(self, price_table, strategy, split_list=None, reward_weight=None):
        """

        :param n_dim: int, default 0
            dimension of x_data
            currently is useless
        :param price_table: list of float
            the closed price corresponding to the x_data
            will be used to evaluate the fitenss

        """

        if split_list is None:
            split_list = [0, len(price_table)]
        if reward_weight is None:
            reward_weight = [1]

        self.split_list = split_list
        self.reward_weight = reward_weight

        self.price_table = price_table
        self.len_price_table = len(price_table)
        self.strategy = strategy

        self.explict_fiteness_func = self.make_explict_func()
        self.est = None
        self.max_samples = 1.0

    def make_explict_func(self):

        reward_func = self.strategy.get_reward
        split_list = self.split_list
        price_table = self.price_table
        reward_weight = self.reward_weight

        def explicit_fitness(y, y_pred, sample_weight):

            y_pred[y_pred == 0] = 1
            total_bool_sample_weight = np.array(sample_weight, dtype=bool)
            result = 0

            total_data = split_list[-1] - split_list[0]
            ratio = np.around(np.sum(total_bool_sample_weight) / total_data, decimals=1)
            is_training_data = total_bool_sample_weight[0]

            for i in range(len(split_list) - 1):
                start = split_list[i]
                end = split_list[i + 1]
                n_data = int(ratio * (end - start))

                _y_pred = np.array(y_pred[start:end])
                _price_table = np.array(price_table[start:end])

                if is_training_data:
                    indices = np.array(range(n_data))
                else:
                    indices = np.array(range(end - n_data, end))

                result += reward_weight[i] * reward_func(_y_pred, _price_table, indices)

                # if is_training_data:
                #     print(indices)
                #     op_arr = CrossBarStrategy.get_op_arr(y_pred)
                #     profit_arr = CrossBarStrategy.get_profit_by_op(op_arr, price_table, indices)
                #     print(np.sum(profit_arr))
                #     print(result)
                # for i in range(len(y_pred)):
                #     print(y_pred[i], op_arr[i], _price_table[i], profit_arr[i])
                #
                # print(result, print(np.sum(profit_arr)))
                # input()
            return result

        return explicit_fitness

    def get_best_oob_est(self):

        _programs = self.est._programs[0]
        best_est = None
        best_oob_fintess = 0
        for _program in _programs:
            if not _program is None:
                if self.greater_is_better:
                    if best_oob_fintess < _program.oob_fitness_:
                        best_oob_fintess = _program.oob_fitness_
                        best_est = _program
                else:
                    if best_oob_fintess > _program.oob_fitness_:
                        best_oob_fintess = _program.oob_fitness_
                        best_est = _program
        return best_est

    def set_params(self, population_size=500, generations=10, tournament_size=20, stopping_criteria=10, p_crossover=0.7, p_subtree_mutation=0.1,
                   greater_is_better=True,
                   p_hoist_mutation=0.05, p_point_mutation=0.1, verbose=1, parsimony_coefficient=0.01, function_set=None, max_samples=1.0):
        self.population_size = population_size
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.verbose = verbose
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.greater_is_better = greater_is_better
        self.tournament_size = tournament_size

        if function_set is None:
            function_set = ['add', 'sub', 'mul', 'div', 'sin']

        self.function_set = function_set

    def fit(self, x_data):
        if not hasattr(self, 'function_set'):
            print("Automatically initilizing....")
            self.set_params()

        metric = make_fitness(self.explict_fiteness_func, greater_is_better=self.greater_is_better)

        # here the random_state is set to be 223 to ensure
        # the cut-split not the sampling split

        self.est = SymbolicRegressor(population_size=self.population_size, tournament_size=self.tournament_size,
                                     generations=self.generations, stopping_criteria=self.stopping_criteria,
                                     p_crossover=self.p_crossover, p_subtree_mutation=self.p_subtree_mutation,
                                     p_hoist_mutation=self.p_hoist_mutation, p_point_mutation=self.p_point_mutation,
                                     metric=metric, max_samples=self.max_samples,
                                     function_set=self.function_set,
                                     verbose=self.verbose, parsimony_coefficient=self.parsimony_coefficient)

        print("x_data.shape[0]", x_data.shape[0])
        self.est.fit(x_data, np.arange(x_data.shape[0]))

        return self.est

    def predict(self, x_data):
        return self.est.predict(x_data)

    def save(self, folder_path, to_save_generation=True):
        """Due to pickle not supporting saving ctype,
              so we have to save the best program  without metric function rather than whole regressor

          :param folder_path: str
                where to save the best est
          """

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_path = "{fd}/model.pkl".format(fd=folder_path)
        plot_path = "{fd}/expression.png".format(fd=folder_path)
        formula_path = "{fd}/mc_formula.txt".format(fd=folder_path)

        best_program = self.est._program
        graph = pydotplus.graphviz.graph_from_dot_data(best_program.export_graphviz())
        graph.write_png(plot_path)

        best_program.metric = None

        setattr(best_program, "predict", best_program.execute)

        with open(model_path, "wb") as f:
            cPickle.dump(best_program, f)

        # save as mc_formula
        formula = best_program.__str__()
        mc_formula = YCInterpreter.mc_interprete(formula)
        with open(formula_path, "w") as f:
            f.write("f[1]=" + mc_formula + ";")

        if to_save_generation:
            self._save_generation("{fd}/generation.pkl".format(fd=folder_path))

    def _save_generation(self, generation_path):
        _programs = self.est._programs[0]
        __programs = []
        for _program in _programs:
            if not _program is None:
                _program.metric = None
                setattr(_program, "predict", _program.execute)

                __programs.append(_program)

        __programs = sorted(__programs, key=lambda x: x.fitness_, reverse=True)

        with open(generation_path, "wb") as f:
            cPickle.dump(__programs, f)
