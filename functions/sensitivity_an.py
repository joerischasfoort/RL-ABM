from qe_model import *
from init_objects import *
from functions.helpers import *


def simulate_params_efast(NRUNS, parameter_set, fixed_parameters):
    """
    Simulate the model twice for different parameter sets. Once with BLR and once without. Record the difference in volatility.
    :param NRUNS: integer amount of Monte Carlo simulations
    :param parameter_set: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed
    :return: numpy array of average stylized facts outcome values for all parameter combinations
    """
    av_diff_BLR = []

    for parameters in parameter_set:
        # combine individual parameters with fixed parameters
        params = fixed_parameters.copy()
        params.update(parameters)

        scenarios = [None, 'BLR']

        # simulate the model

        av_volatility = {}
        for scenario in scenarios:
            trdrs = []
            orbs = []
            central_banks = []

            prices = []
            fundamentals = []
            for seed_nb in range(NRUNS):
                traders_nb, central_bank_nb, orderbook_nb = init_objects(params, seed_nb)
                traders_nb, central_bank_nb, orderbook_nb = qe_model(traders_nb, central_bank_nb, orderbook_nb,
                                                                     params, scenario=scenario, seed=seed_nb)
                central_banks.append(central_bank_nb)
                trdrs.append(traders_nb)
                orbs.append(orderbook_nb)

            prices = pd.DataFrame([orbs[run].tick_close_price for run in range(NRUNS)]).transpose()
            fundamentals = pd.DataFrame([orbs[run].fundamental for run in range(NRUNS)]).transpose()
            pfs = (prices / fundamentals)[:-1]
            # calculate volatility

            av_volatility[scenario] = np.mean(pfs.std()) / np.mean(pfs)

        av_diff_BLR.append(np.mean(av_volatility[scenarios[1]] - av_volatility[scenarios[0]]))

    return av_diff_BLR
