from initialize_model import *
from model import *
import time
import matplotlib.pyplot as plt
import seaborn as sns

start_time = time.time()

# # 1 setup parameters
parameters = {'trader_sample_size': 10,
              'n_traders': 50,
              'init_stocks': 81,
              'ticks': 606,
              'fundamental_value': 1112.2356754564078,
              'std_fundamental': 0.036106530849401956,
              'base_risk_aversion': 0.7,
              'spread_max': 0.004087,
              'horizon': 212,
              'std_noise': 0.05149715506250338,
              'w_random': 1.0,
              'mean_reversion': 0.0,
              'fundamentalist_horizon_multiplier': 1.0,
              'strat_share_chartists': 0.0,
              'mutation_intensity': 0.0,
              'average_learning_ability': 0.0,
              'trades_per_tick': 1,
              'verbose': False}

# 2 initialise model objects
traders, orderbook, market_maker = init_objects(parameters, seed=0)

# 3 simulate model
traders, orderbook, market_maker = ABM_model(traders, orderbook, market_maker, parameters, seed=0)


print("The simulations took", time.time() - start_time, "to run")

# Plot market maker's wealth over time
sns.set_style("darkgrid")
plt.plot(market_maker.metrics['wealth'])
plt.ylabel('Market maker wealth')
plt.show()