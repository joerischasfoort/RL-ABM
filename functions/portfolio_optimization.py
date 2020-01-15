import numpy as np
import sys
import pandas as pd


def portfolio_optimization(trader, day):
    """
    Calculate optimal weights of assets given covariance, return expectations and risk aversion
    :param trader: trader which tries to optimize its portfolio
    :param day: period at which the optimization takes place
    :return:
    """
    # create a copy of the covariance matrix of the funds
    covariance_assets = trader.var.covariance_matrix.copy()

    expected_return_assets = np.zeros((len(covariance_assets)))

    # compute the risk aversion matrix
    risk_aversion_mat = pd.DataFrame((np.zeros((2, 2)) + trader.par.risk_aversion) , index=covariance_assets.columns, columns=covariance_assets.columns)

    # multiply covariance with asset specific risk aversion
    covariance_assets = np.multiply(covariance_assets, risk_aversion_mat)
    original_cov = np.array(covariance_assets)

    # Create a 1D numpy array with one expected returns per asset
    for i, a in enumerate(covariance_assets.columns.values):
        expected_return_assets[i] = trader.exp.returns[a]

    # adding the budget constraint to the covariance matrix - to solve the model using linear algebra
    # Adding a row with ones
    aux_cov = np.concatenate((covariance_assets, np.ones((1, len(covariance_assets)))), axis=0)

    # Adding a column with ones
    aux_cov = np.concatenate((aux_cov, np.ones((len(aux_cov), 1))), axis=1)
    aux_cov[len(aux_cov) - 1, len(aux_cov) - 1] = 0

    # adding the budget constraint to the return vector - to solve the model using linear algebra
    aux_x = np.append(expected_return_assets, 0)
    aux_y = np.zeros(len(aux_x))
    aux_y[len(aux_x) - 1] = 1

    o_aux_x = aux_x.copy()
    o_aux_y = aux_y.copy()
    o_aux_cov = aux_cov.copy()

    kt_count =0
    test_KT = np.zeros(len(aux_x)-1)
    while sum(test_KT) != len(test_KT):
        if kt_count > (len(aux_x)-1):
            print("day ", day)

        # compute matrix inverse
        try:
            inv_aux_cov = np.linalg.inv(aux_cov)
        except:
            print('error')

        inv_aux_cov = np.linalg.inv(aux_cov)
        aux_c = np.matmul(inv_aux_cov, aux_x)
        aux_d = np.matmul(inv_aux_cov, aux_y)

        # solving for optimal weights
        weights = aux_c + aux_d

        # Start of algorithm that takes out shorted assets
        test = weights[:-1] < 0

        while sum(test) > 0:
            for i in range(len(aux_cov)-1):
                for j in range(len(aux_cov)):
                    if weights[i] < 0 and i != j:
                        aux_cov[i, j] = 0
                    if weights[i] < 0 and i == j:
                        aux_cov[i, j] = 1
            for i in range(len(aux_x) - 1):
                if weights[i] < 0:
                    aux_x[i] = 0
                    aux_y[i] = 0

            # compute matrix inverse
            inv_aux_cov = np.linalg.inv(aux_cov)
            aux_c = np.matmul(inv_aux_cov, aux_x)
            aux_d = np.matmul(inv_aux_cov, aux_y)

            weights = aux_c + aux_d
            test = weights[:-1] < 0

        aux_e = np.zeros(len(aux_c) - 1)
        aux_f = np.zeros(len(aux_c) - 1)
        for i in range(len(aux_e)):
            aux_e[i] = expected_return_assets[i] - sum([original_cov[i, j] * aux_c[j] for j in range(len(aux_e))]) - aux_c[-1]
            aux_f[i] = sum([original_cov[i, j] * aux_d[j] for j in range(len(aux_e))]) + aux_d[-1]

        aux_kt_pd = aux_e - aux_f  # partial derivatives
        for i in range(len(aux_kt_pd)):
            if weights[i] > 0:
                test_KT[i] = 1
            else:
                test_KT[i] = aux_kt_pd[i] <= sys.float_info.epsilon
        # if Kuhn-Tucker conditions are not fulfilled put the asset with the highest partial derivative (marginal utility) back
        if sum(test_KT) < len(test_KT):
            kt_count = kt_count + 1
            test_KT_inv = 1-test_KT
            aux2_KT_pd = aux_kt_pd * test_KT_inv
            aux2_KT_pd = aux2_KT_pd.tolist()
            max_pd = max(aux2_KT_pd)
            i = aux2_KT_pd.index(max_pd)
            if test_KT[i] == 0:
                aux_x[i] = o_aux_x[i]
                aux_y[i] = o_aux_y[i]
                for j in range(len(aux_cov)):
                    aux_cov[i, j] = o_aux_cov[i, j]

    # change here the output to a list of two weights (stocks, cash)
    output = {}

    for i, a in enumerate(covariance_assets.columns.values):
        output[a] = weights[i]

    return output
