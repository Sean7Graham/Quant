import numpy as np
import seaborn as sm


# NOTE CRITICAL LEVEL HAS BEEN SET TO 5% FOR COINTEGRATION TEST
def find_cointegrated_pairs(dataframe, critial_level=0.05):
    n = dataframe.shape[1]  # the length of dateframe
    pvalue_matrix = np.ones((n, n))  # initialize the matrix of p
    keys = dataframe.columns  # get the column names
    pairs = []  # initilize the list for cointegration
    for i in range(n):
        for j in range(i + 1, n):  # for j bigger than i
            stock1 = dataframe[keys[i]]  # obtain the price of "stock1"
            stock2 = dataframe[keys[j]]  # obtain the price of "stock2"
            result = sm.tsa.stattools.coint(stock1, stock2)  # get conintegration
            pvalue = result[1]  # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level:  # if p-value less than the critical level
                pairs.append(
                    (keys[i], keys[j], pvalue)
                )  # record the contract with that p-value
    return pvalue_matrix, pairs

    ## Examine to make sure that the criticial test is performed as desired
