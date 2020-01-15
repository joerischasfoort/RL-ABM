import numpy as np


def gini(array):
    """
    Calculate the Gini coefficient of a numpy array.
    All credits to Olivia Guest @ https://github.com/oliviaguest/gini
    based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    array = array.flatten()

    # make sure values are not negative
    if np.amin(array) < 0:
        array -= np.amin(array)
        print('Negative values founds, check calculation')

    # slightly offset values of 0
    array += 0.0000001

    # sort values
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0] # number of array elements

    gini_coefficient = ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    return gini_coefficient
