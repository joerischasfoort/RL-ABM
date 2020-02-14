from initialize_model import *
from model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {'trader_sample_size': 10,
              'n_traders': 50,
              'init_stocks': 81,
              'ticks': 50,
              'fundamental_value': 90.0,
              'std_fundamental': 0.0361,
              'base_risk_aversion': 0.7,
              'spread_max': 2.0,
              'horizon': 30,
              'std_noise': 0.05,
              'w_random': 1.0,
              'mean_reversion': 0.0,
              'money_multiplier': 1.2,
              'fundamentalist_horizon_multiplier': 1.0,
              'strat_share_chartists': 0.0,
              'mutation_intensity': 0.0,
              'average_learning_ability': 0.0,
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
traders, orderbook, rl_agent = init_objects(parameters, seed=seed) #TODO, update init objects to yield rl_agent, now it is the market maker

# 3 simulate model
traders, orderbook, rl_agent = seller_model(traders, orderbook, rl_agent, parameters, seed=seed)

this_final_rel_wealth = rl_agent.var.wealth[-1] / rl_agent.var.wealth[0]
final_rel_wealth.append(this_final_rel_wealth)

print(f"final relative wealth of market maker {final_rel_wealth[0]}")