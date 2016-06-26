def duplicator(data, number=10, iters=0):
    """
    A function for duplicating transactions.

    data (array): An array of transactions,
    number (int): The number of rows from the original dataset to be included.
    iters (int): The number of times to duplicate the rows.
    """
    txns = data.ix[:(number-1), 'itemset']
    index = number+1

    for i in range(iters):
        for n in range(number):
            txns[(index)] = txns[n]
            index += 1

    return txns

data = duplicator(df_apriori)