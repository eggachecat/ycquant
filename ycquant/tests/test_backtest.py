from ycquant.yc_backtest import *
import numpy as np

value_list = np.array([80, -32, -32, 38, -60, -6, -63, -114, -110, -94, -114, -96, 57])
price_table = np.array([6036, 5948, 5968, 6099, 6031, 6122, 6050, 5952, 5889, 5836, 5737, 5639, 5729])
indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# op_arr = CrossBarStrategy.get_op_arr(value_list)
# profit = CrossBarStrategy.get_reward(value_list, price_table, indices)

# op_arr = CrossBarStrategy.get_op_arr(value_list)

profit = CrossBarStrategy.get_reward(value_list, price_table, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

print(profit)
