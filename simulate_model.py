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
              'ticks': 601,
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

'''

# 2 initialise model objects
traders, orderbook, market_maker, dumb_market_maker = init_objects(parameters, seed=0)

# 3 simulate model
traders, orderbook, market_maker, dumb_market_maker = ABM_model(traders, orderbook, market_maker, dumb_market_maker, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")


# Plot market maker's wealth over time
sns.set_style("darkgrid")
plt.plot(market_maker.var.wealth)
plt.ylabel('Market maker wealth')
plt.show()

# Plot market maker's wealth over time
sns.set_style("darkgrid")
plt.plot(dumb_market_maker.var.wealth)
plt.ylabel('Dumb market maker wealth')
plt.show()
'''

final_rel_wealth = []
seed = 2

# 2 initialise model objects
traders, orderbook, market_maker = init_objects(parameters, seed=seed)

# 3 simulate model
traders, orderbook, market_maker = ABM_model(traders, orderbook, market_maker, parameters, seed=seed)

this_final_rel_wealth = market_maker.var.wealth[-1] / market_maker.var.wealth[0]
final_rel_wealth.append(this_final_rel_wealth)


print(f"final relative wealth of market maker {final_rel_wealth[0]}")