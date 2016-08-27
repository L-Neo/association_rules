#
# Python 3 Version
#
# This algorithm is adapted from a book on machine learning.
# 
# Book: Machine Learning in Action
# Author: Pete Harrington
# Page: 229

import pandas as pd

def CreateBaseSet(transactions):
    """
    Creates sets of single items from a list of transactions.

    Args:
        transactions (list): A list of transactions. Each transaction is a list of items.

    Returns:
        base_items (list): A list of frozen sets.
    """
    base_items = []

    for itemset in transactions:
        for item in itemset:
            if not [item] in base_items:
                base_items.append([item])
    base_items = list(map(frozenset, base_items))           

    return base_items

def ScanSets(transactions, itemset, min_support):
    """
    Scans the sets, calculates the support, and returns itemsets that meet the minimum support. 

    Args:
        transactions (list): A list of transactions. Each transaction is a list of items.

    Returns:
        itemsets (list): A list of items that meet the minimum support.
        itemsets_support (dictionary): A dictionary of itemsets mapped to their support values.
    """
    tmp = {}

    for transaction in transactions:
        for item in itemset:
            if item.issubset(transaction):
                if tmp.get(item) == None:
                    tmp[item] = 1
                else:
                    tmp[item] += 1

    length = len(transactions)
    itemsets = []
    itemsets_support = {}

    for itemset in tmp:
        support = tmp[itemset] / float(length)
        if support >= min_support:
            itemsets.append(itemset)
            itemsets_support[itemset] = support

    return itemsets, itemsets_support

def GenerateSets(itemsets):
    """
    Generates new itemsets by combining each itemset with all other itemsets.

    Args:
        itemsets (list): A list of itemsets to be used for generating new itemsets.

    Returns:
        new_set (list): A list of new sets generated from the original list of itemsets.
    """
    new_set = []
    set_length = len(itemsets)

    for index, item in enumerate(itemsets):
        for nextitem in itemsets[index+1:]:
            union_set = item|nextitem
            if union_set not in new_set:
                new_set.append(union_set)
    return new_set

def Apriori(transactions, min_support=0.01):
    """
    Implements the apriori algorithm on a list of transactions.

    Args:
        transactions (list): A list of transactions. Each transaction is a list of items.
        min_support (float): The minimum support required for each itemset.

    Returns:
        itemsets_support (dataframe): A dataframe with the itemsets and their corresponding support values.
    """
    min_support = float(min_support)

    baseset = CreateBaseSet(transactions)
    remaining_items, itemsets_support = ScanSets(transactions, baseset, min_support)
    all_itemsets = list(remaining_items)

    while len(remaining_items) > 0:
        new_sets = GenerateSets(remaining_items)
        remaining_items, new_itemsets_support = ScanSets(transactions, new_sets, min_support)
        itemsets_support.update(new_itemsets_support)
        all_itemsets += remaining_items

    itemsets_support = pd.DataFrame.from_dict(itemsets_support, orient='index').reset_index()
    itemsets_support.columns = ['itemsets','support']
    itemsets_support.sort_values(by='support', ascending=False, inplace=True)
    itemsets_support.reset_index(inplace=True, drop=True)

    return itemsets_support