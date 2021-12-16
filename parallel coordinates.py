import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

deter= pd.read_csv("./Results_To_Plot/pareto_deter.csv", sep=";", usecols=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_fen"])
deter["model"]='deter'
#print(deter.head())
print(deter.shape)
nomass= pd.read_csv("./Results_To_Plot/pareto_nomass.csv", sep=";", usecols=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_fen"])
nomass["model"]="nomass"
nomass=nomass.dropna(axis=1)
print(nomass.shape)
data = pd.concat([deter,nomass],ignore_index=True)
data.to_csv("global_pareto.csv",index=False)
#print(data.head())
#print(data.shape)

'''parallel_coordinates(data,

                    class_column="model",color=['r','b'])
plt.show()'''
import plotly.express as px
'''fig = px.parallel_coordinates(data,
                            dimensions=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen","besoins_chauff_kwh_m2","inconfort_heures","couts_euros_m2"],
                            color="model")
fig.show()'''
fig = px.parallel_categories(deter,
                            dimensions=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen"])
fig.show()

'''df_param=data.groupby(["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen"]).size()
print(df_param.value_counts())

df_param.to_csv("duplicates_param.csv")'''
'''df_param=data.groupby(["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen"])
df_param=df_param[df_param.size()==1]
print(df_param.head())'''