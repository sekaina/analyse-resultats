import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
header_list=["ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_fenetre"]
df_deter=pd.read_csv('./Results_To_Plot/pareto_param_gen99_deter.csv', names=header_list)
df_nomass=pd.read_csv('./Results_To_Plot/pareto_param_gen99_nomass.csv', names=header_list)
indiv_deter=len(df_deter)
indiv_nomass=len(df_nomass)
print(df_deter['type_fenetre'].value_counts())
print(df_nomass['type_fenetre'].value_counts())
print(df_deter['ep_murs_ext'].value_counts())
print(df_nomass['ep_murs_ext'].value_counts())
bins_list=np.linspace(10,50,5)
(df_deter['type_fenetre'].value_counts()*100/indiv_deter).plot.bar()
(df_nomass['type_fenetre'].value_counts()*100/indiv_nomass).plot.bar()
#df_deter['ep_murs_ext'].value_counts().plot.bar()

#type_fen=pd.merge(df_deter['type_fenetre'].value_counts()*100/indiv_deter,df_nomass['type_fenetre'].value_counts()*100/indiv_nomass)
'''
plt.hist(df_deter['ep_murs_ext'],bins_list,alpha=0.5,label="deter")
plt.hist(df_nomass['ep_murs_ext'],bins_list,alpha=0.5,label="nomass")
plt.legend()

'''
#print(type_fen)
plt.show()