import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

'''with open('./Results_To_Plot/pareto_obj_gen99_nomass.csv', 'r') as f:
    Pareto_objective_functions_nomass=np.array(list(csv.reader (f, delimiter=',')))
costs_nomass=Pareto_objective_functions_nomass.astype('float64')
with open('./Results_To_Plot/pareto_obj_gen99_deter.csv', 'r') as f:
    Pareto_objective_functions_deterministic=np.array(list(csv.reader (f, delimiter=',')))
Pareto_objective_functions_deterministic=Pareto_objective_functions_deterministic.astype('float64')'''

df=pd.read_excel("./Results_To_Plot/exhaustive_sans_surventilation_all_combinaisons.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
df_array=df[["f1","f2","f3"]].to_numpy()
df_costs_array=df_array.astype('float64')
is_efficient=is_pareto_efficient_simple(df_costs_array)
df["efficient"]=is_efficient
#print(df)
df[df["efficient"]].to_excel("exhaustive_sans_surventilation_pareto.xlsx",index=False)
pareto_new=0
'''for i, a in enumerate(pareto_tot):
    if a :
        pareto_new+=1
print(pareto_new)'''
