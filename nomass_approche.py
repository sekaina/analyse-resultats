from numpy import float64
import pandas as pd
nomass= pd.read_csv("./Results_To_Plot/pareto_nomass.csv", sep=";")
duplicates= pd.read_csv("duplicates.csv", sep=",")
duplicates_with_fitnesses=nomass.merge(duplicates, how='inner', on=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_vitrage"])
print(duplicates_with_fitnesses)
duplicates_with_fitnesses.to_csv("duplicates_with_fitnesses.csv", index=False)