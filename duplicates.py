import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

deter= pd.read_csv("./Results_To_Plot/pareto_deter.csv", sep=";", usecols=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_vitrage"])
#deter["model"]='deter'
#print(deter.head())
#print(deter.shape)
nomass= pd.read_csv("./Results_To_Plot/pareto_nomass.csv", sep=";", usecols=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_vitrage"])
#nomass["model"]="nomass"
nomass=nomass.dropna(axis=1)
#print(nomass.shape)
'''data = pd.concat([deter,nomass]).drop_duplicates().reset_index(drop=True)
data.to_csv("global_pareto.csv")#,index=False
#print(data.head())
#print(data.shape)

#deter_non_nomass=pd.concat([nomass, data]).drop_duplicates().reset_index(drop=True)
deter_non_nomass=deter.merge(nomass, how="outer", on=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_fen"])'''
'''df_param=data.groupby(["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen"]).size()
print(df_param.value_counts())
df_param.to_csv("duplicates_param.csv")'''
'''deter_non_nomass.to_csv("deter_non_nomass.csv")
print(deter_non_nomass.shape)'''


'''df_param=data.groupby(["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm",
                            "type_fen"])
df_param=df_param[df_param.size()==1]
print(df_param.head())'''
count=0
duplicates=deter.merge(nomass, how="inner")#, on=["ep_murs_ext_cm","ep_plancher_haut_cm","ep_plancher_bas_cm","type_fen"]
duplicates.to_csv("duplicates.csv", index=False)
#deter_non_nomass=pd.concat([deter,duplicates]).drop_duplicates()#.reset_index(drop=True)
#print(deter_non_nomass.shape)
'''for i in range(len(deter)):
    for j in range (len(deter_non_nomass)):
        if ((deter.loc[i,"ep_murs_ext_cm"]!=deter_non_nomass.loc[j,"ep_murs_ext_cm"])
        &(deter.loc[i,"ep_plancher_haut_cm"]!=deter_non_nomass.loc[j,"ep_plancher_haut_cm"])
        &(deter.loc[i,"ep_plancher_bas_cm"]!=deter_non_nomass.loc[j,"ep_plancher_bas_cm"])
        &(deter.loc[i,"type_fen"]!=deter_non_nomass.loc[j,"type_fen"])):
            count+=1
print(count/4)'''
def get_different_rows(source_df, new_df):
    """Returns just the rows from the new dataframe that differ from the source dataframe"""
    merged_df = source_df.merge(new_df, indicator=True, how='outer')
    changed_rows_df = merged_df[merged_df['_merge'] == 'right_only']
    return changed_rows_df.drop('_merge', axis=1)
print(get_different_rows(duplicates,deter))
df=get_different_rows(duplicates,deter)
#df.to_csv("deter_non_nomass.csv", index=False)
select=nomass[nomass["ep_murs_ext_cm"].isin(deter)]#.loc[(nomass1["ep_murs_ext_cm"]==duplicates1["ep_murs_ext_cm"])#&(nomass1["ep_plancher_haut_cm"]==duplicates1["ep_plancher_haut_cm"])
                    #&(nomass1["ep_plancher_bas_cm"]==duplicates1["ep_plancher_bas_cm"])&(nomass1["type_fen"]==duplicates1["type_fen"])]
print(select)