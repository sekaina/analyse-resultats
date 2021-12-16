import numpy as np
import matplotlib.pyplot as plt

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
def is_dominated(B,A):
    """donne combien d'individus de A sont dominés par au moins un individu de B"""
    is_dominated = np.zeros(A.shape[0], dtype = bool)
    for i, a in enumerate(A): #prend chaque individu de new
        l_is_dominated=np.zeros(B.shape[0], dtype = bool)
        for j, b in enumerate(B):
            l_is_dominated[j]=np.all(b<=a) #compare avec tous les individus de B, 
                                            #si l'individu de origin domine l'individu de A
        is_dominated[i] = np.any(l_is_dominated)#np.all(np.any(origin>c, axis=1))
    return is_dominated
import csv
import pandas as pd
with open('./Results_To_Plot/pareto_obj_gen99_nomass.csv', 'r') as f:
    Pareto_objective_functions_nomass=np.array(list(csv.reader (f, delimiter=',')))
costs_nomass=Pareto_objective_functions_nomass.astype('float64')
with open('./Results_To_Plot/pareto_obj_gen99_deter.csv', 'r') as f:
    Pareto_objective_functions_deterministic=np.array(list(csv.reader (f, delimiter=',')))
Pareto_objective_functions_deterministic=Pareto_objective_functions_deterministic.astype('float64')
print(costs_nomass.shape)
df_deter_non_nomass=pd.read_csv("./Results_To_Plot/deter_non_nomass.csv", header=None, sep=';')
df_array=df_deter_non_nomass.to_numpy()
df_array=df_array.astype('float64')
print(df_array.shape)
df_tot=np.append(costs_nomass, df_array, axis=0)
print (df_tot.shape)
c_nomass=np.array([[12.19,125.7, 190.11]])
c_deter=np.array([[5.1, 134.35 , 170]])
#print(np.any(costs_nomass<c1))
#print(costs_nomass<c_nomass)
#print (Pareto_objective_functions_deterministic<c_deter)
#print(compare_is_pareto_efficient_dumb(costs_nomass,df_array))
deter_dominated_by_nomass=is_dominated(costs_nomass,df_array)
nomass_dominated_by_deter=is_dominated(df_array,costs_nomass)
print(nomass_dominated_by_deter)
are_dominated=0
dominates=0
for i, a in enumerate(deter_dominated_by_nomass):
    if a :
        are_dominated+=1
for i, a in enumerate(nomass_dominated_by_deter):
    if a :
        dominates+=1
print(deter_dominated_by_nomass.shape[0])
print(nomass_dominated_by_deter.shape[0])
print(are_dominated, dominates)
pareto_tot=is_pareto_efficient(df_tot)
pareto_new=0
for i, a in enumerate(pareto_tot):
    if a :
        pareto_new+=1
print(pareto_new)
'''A=np.array([[1,2, 3],[0,1,2],[7,8,9]])
B=np.array([[0.5,1.5, 2.5],[1,7,6],[5,7,9]])
print(is_dominated(B,A))
print(B<=A)'''