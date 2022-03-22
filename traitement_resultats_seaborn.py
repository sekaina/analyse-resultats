import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import openturns as ot
def SRC(X,Y):
    from scipy import stats
    import statsmodels.api as sm
    Y_norm= pd.Series(stats.zscore(Y), name=Y.name)
    X_norm = pd.DataFrame(stats.zscore(X))
    X_norm.columns=X.columns
    #print(round(Y_norm.mean(axis=0),5))
    modstd=sm.OLS(Y_norm,X_norm)
    modstd_res=modstd.fit()
    print(modstd_res.summary())
    coeff = modstd_res.params
    #coeff = coeff.iloc[(coeff.abs()*-1.0).argsort()]
    #sns.barplot(x=coeff.values, y=coeff.index, orient='h')
    #plt.show()
    print(modstd_res.params.values)
    return modstd_res.params
def SRRC(X,Y):
    X=X.to_numpy()
    Y=Y.to_numpy()
    m=Y.shape[0]
    Y.shape=(m,1)
    inputDesign = ot.Sample(X)
    outputDesign = ot.Sample(Y)
    srrc_indices = ot.CorrelationAnalysis.SRRC(X, Y,normalize=True)
    return np.array(srrc_indices).tolist() 
def plot_SRRC(df1,df2,df3):
    X1=df1[["x1","x2","x3","x4"]]
    Y11=df1["f1"]
    Y12=df1["f2"]
    Y13=df1["f3"]
    X2=df2[["x1","x2","x3","x4"]]
    Y21=df2["f1"]
    Y22=df2["f2"]
    Y23=df2["f3"]
    X3=df3[["x1","x2","x3","x4"]]
    Y31=df3["f1"]
    Y32=df3["f2"]
    Y33=df3["f3"]
    A11=SRRC(X1,Y11)
    A12=SRRC(X1,Y12)
    A13=SRRC(X1,Y13)
    A21=SRRC(X2,Y21)
    A22=SRRC(X2,Y22)
    A23=SRRC(X2,Y23)
    A31=SRRC(X3,Y31)
    A32=SRRC(X3,Y32)
    A33=SRRC(X3,Y33)
    A11=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A11})
    A11["Modèle"]="Scénarii fixes"
    A21=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A21})
    A21["Modèle"]="NoMASS"
    A31=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A31})
    A31["Modèle"]="NoMASS approché"
    A=pd.concat([A11,A21,A31], axis=0, join='outer', ignore_index=True)
    A12=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A12})
    A12["Modèle"]="Scénarii fixes"
    A22=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A22})
    A22["Modèle"]="NoMASS"
    A32=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A32})
    A32["Modèle"]="NoMASS approché"
    B=pd.concat([A12,A22,A32], axis=0, join='outer', ignore_index=True)
    A13=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A13})
    A13["Modèle"]="Scénarii fixes"
    A23=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A23})
    A23["Modèle"]="NoMASS"
    A33=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRRC": A33})
    A33["Modèle"]="NoMASS approché"
    C=pd.concat([A13,A23,A33], axis=0, join='outer', ignore_index=True)
    fig, ax = plt.subplots(nrows=1,ncols=3)
    a=sns.barplot(x="SRRC", y="Variable", orient='h',data=A,hue='Modèle',ax=ax[0])
    b=sns.barplot(x="SRRC", y="Variable", orient='h',data=B,hue='Modèle',ax=ax[1])
    c=sns.barplot(x="SRRC", y="Variable", orient='h',data=C,hue='Modèle',ax=ax[2])
    #b.tick_params(labelsize=5)
    b.legend_.remove()
    a.legend_.remove()
    a.set_title("f1")
    b.set_title("f2")
    c.set_title("f3")
    plt.show()
def plot_SRC(df1,df2,df3):
    X1=df1[["x1","x2","x3","x4"]]
    Y11=df1["f1"]
    Y12=df1["f2"]
    Y13=df1["f3"]
    X2=df2[["x1","x2","x3","x4"]]
    Y21=df2["f1"]
    Y22=df2["f2"]
    Y23=df2["f3"]
    X3=df3[["x1","x2","x3","x4"]]
    Y31=df3["f1"]
    Y32=df3["f2"]
    Y33=df3["f3"]
    A11=SRC(X1,Y11)
    A12=SRC(X1,Y12)
    A13=SRC(X1,Y13)
    A21=SRC(X2,Y21)
    A22=SRC(X2,Y22)
    A23=SRC(X2,Y23)
    A31=SRC(X3,Y31)
    A32=SRC(X3,Y32)
    A33=SRC(X3,Y33)
    A11=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A11.values)})
    A11["Modèle"]="Scénarii fixes"
    A21=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A21.values)})
    A21["Modèle"]="NoMASS"
    A31=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A31.values)})
    A31["Modèle"]="NoMASS approché"
    A=pd.concat([A11,A21,A31], axis=0, join='outer', ignore_index=True)
    A12=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A12.values)})
    A12["Modèle"]="Scénarii fixes"
    A22=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A22.values)})
    A22["Modèle"]="NoMASS"
    A32=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A32.values)})
    A32["Modèle"]="NoMASS approché"
    B=pd.concat([A12,A22,A32], axis=0, join='outer', ignore_index=True)
    A13=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A13.values)})
    A13["Modèle"]="Scénarii fixes"
    A23=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A23.values)})
    A23["Modèle"]="NoMASS"
    A33=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A33.values)})
    A33["Modèle"]="NoMASS approché"
    C=pd.concat([A13,A23,A33], axis=0, join='outer', ignore_index=True)
    fig, ax = plt.subplots(nrows=1,ncols=3)
    a=sns.barplot(x="SRC", y="Variable", orient='h',data=A,hue='Modèle',ax=ax[0])
    b=sns.barplot(x="SRC", y="Variable", orient='h',data=B,hue='Modèle',ax=ax[1])
    c=sns.barplot(x="SRC", y="Variable", orient='h',data=C,hue='Modèle',ax=ax[2])
    #b.tick_params(labelsize=5)
    b.legend_.remove()
    a.legend_.remove()
    a.set_title("f1")
    b.set_title("f2")
    c.set_title("f3")
    #ax.barh([y_pos,y_pos], [list(A11.values),list(A21.values)], align='center')
    #ax.set_yticks(y_pos, labels=people)
    #ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')
    plt.show()
def plot_SRC_all(df1,df2):
    X1=df1[["x1","x2","x3","x4"]]
    Y11=df1["f1"]
    Y12=df1["f2"]
    Y13=df1["f3"]
    X2=df2[["x1","x2","x3","x4"]]
    Y21=df2["f1"]
    Y22=df2["f2"]
    Y23=df2["f3"]
    A11=SRC(X1,Y11)
    A12=SRC(X1,Y12)
    A13=SRC(X1,Y13)
    A21=SRC(X2,Y21)
    A22=SRC(X2,Y22)
    A23=SRC(X2,Y23)
    A11=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A11.values)})
    A11["Modèle"]="Scénarii fixes"
    A21=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A21.values)})
    A21["Modèle"]="NoMASS"
    A=pd.concat([A11,A21], axis=0, join='outer', ignore_index=True)
    A12=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A12.values)})
    A12["Modèle"]="Scénarii fixes"
    A22=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A22.values)})
    A22["Modèle"]="NoMASS"
    B=pd.concat([A12,A22], axis=0, join='outer', ignore_index=True)
    A13=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A13.values)})
    A13["Modèle"]="Scénarii fixes"
    A23=pd.DataFrame({"Variable": ["x1","x2","x3","x4"], "SRC": list(A23.values)})
    A23["Modèle"]="NoMASS"
    C=pd.concat([A13,A23], axis=0, join='outer', ignore_index=True)
    fig, ax = plt.subplots(nrows=1,ncols=3)
    a=sns.barplot(x="SRC", y="Variable", orient='h',data=A,hue='Modèle',ax=ax[0])
    b=sns.barplot(x="SRC", y="Variable", orient='h',data=B,hue='Modèle',ax=ax[1])
    c=sns.barplot(x="SRC", y="Variable", orient='h',data=C,hue='Modèle',ax=ax[2])
    #b.tick_params(labelsize=5)
    b.legend_.remove()
    a.legend_.remove()
    a.set_title("f1")
    b.set_title("f2")
    c.set_title("f3")
    #ax.barh([y_pos,y_pos], [list(A11.values),list(A21.values)], align='center')
    #ax.set_yticks(y_pos, labels=people)
    #ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')
    plt.show()
def all_generations():
    df=pd.DataFrame(columns=list('ABC'))
    for i in range(1,100):
        dfi= pd.read_csv('./Results_To_Plot/monitoring_nomass/pareto_obj_gen'+str(i)+'.csv', header=None,names=list('ABC'))
        df=pd.concat([df,dfi],ignore_index=True)
    df.to_csv("pareto_obj_all_gen_nomass.csv",index=False)
def N_solutions_pareto_per_generation():
    L=[]
    for i in range(1,100):
        dfi= pd.read_csv('./Results_To_Plot/monitoring_deter/pareto_obj_gen'+str(i)+'.csv', header=None)
        #L.append(dfi[0].count)
        print(dfi[0].size)
def plot_density(df1,df2,df3):
    fig,axes=plt.subplots(nrows=7,ncols=1)
    sns.histplot(df1["f1"],ax=axes[0],color='b', label="SF avec surventilation")
    sns.histplot(df3["f1"],ax=axes[0],color='grey', label="SF sans surventilation")
    sns.histplot(df2["f1"],ax=axes[0],color='r', label="NoMASS")
    sns.kdeplot(df1["f2"],ax=axes[1],color='b',shade=True)
    sns.kdeplot(df2["f2"],ax=axes[1],color='r',shade=True)
    sns.kdeplot(df3["f2"],ax=axes[1],color='grey',shade=True)
    sns.kdeplot(df1["f3"],ax=axes[2],color='b',shade=True)
    sns.kdeplot(df2["f3"],ax=axes[2],color='r',shade=True)
    sns.kdeplot(df3["f3"],ax=axes[2],color='grey',shade=True)
    sns.kdeplot(df1["x1"],ax=axes[3],color='b',shade=True)
    sns.kdeplot(df2["x1"],ax=axes[3],color='r',shade=True)
    sns.kdeplot(df3["x1"],ax=axes[3],color='grey',shade=True)
    sns.kdeplot(df1["x2"],ax=axes[4],color='b',shade=True)
    sns.kdeplot(df2["x2"],ax=axes[4],color='r',shade=True)
    sns.kdeplot(df3["x2"],ax=axes[4],color='grey',shade=True)
    sns.kdeplot(df1["x3"],ax=axes[5],color='b',shade=True)
    sns.kdeplot(df2["x3"],ax=axes[5],color='r',shade=True)
    sns.kdeplot(df3["x3"],ax=axes[5],color='grey',shade=True)
    sns.kdeplot(df1["x4"],ax=axes[6],color='b',shade=True)
    sns.kdeplot(df2["x4"],ax=axes[6],color='r',shade=True)
    sns.kdeplot(df3["x4"],ax=axes[6],color='grey',shade=True)
    fig.legend()
    plt.show()
def plot_pareto_f1_f3_SF(df,df2,df3,case=1):
    x = df["f1"]#chauffage
    y = df["f2"] #inconfort
    z = df["f3"] #cout
    x2 = df2["f1"]
    y2 = df2["f2"] 
    z2 = df2["f3"] 
    x3 = df3["f1"]#chauffage
    y3 = df3["f2"] #inconfort
    z3 = df3["f3"] #cout
    fig, ax = plt.subplots()
    if case==1:
        zs = np.concatenate([z, z2,z3], axis=0)
        min_, max_ = zs.min(), zs.max()
        plot=plt.scatter(x, y, c=z, marker="o", label="SF avec surventilation")
        plt.clim(min_, max_)
        plt.scatter(x3, y3, c=z3, marker="P", label="SF sans surventilation")
        plt.clim(min_, max_)
        plt.scatter(x2, y2, c=z2, marker="v", label="NoMASS")
        plt.clim(min_, max_)
        ax.set_ylabel("Heures d'inconfort")
        ax.set_xlabel("Besoins de chauffage kWh/(m2.an)")
        #plt.legend(*plot.legend_elements("sizes", num=6))       
        fig.colorbar(plot,label="Cout actualisé en euros")
        ax.legend()
    if case==2:
        zs = np.concatenate([y, y2,y3], axis=0)
        min_, max_ = zs.min(), zs.max()
        plot=plt.scatter(x, z, c=y, marker="o", label="SF avec surventilation")
        plt.clim(min_, max_)
        plt.scatter(x3, z3, c=y3, marker="P", label="SF sans surventilation")
        plt.clim(min_, max_)
        plt.scatter(x2, z2, c=y2, marker="v", label="NoMASS")
        plt.clim(min_, max_)
        ax.set_ylabel("Cout actualisé en euros")
        ax.set_xlabel("Besoins de chauffage kWh/(m2.an)")
        #plt.legend(*plot.legend_elements("sizes", num=6))       
        fig.colorbar(plot,label="Heures d'inconfort")
        ax.legend()
    if case==3:
        zs = np.concatenate([x, x2,x3], axis=0)
        min_, max_ = zs.min(), zs.max()
        plot=plt.scatter(y, z, c=x, marker="o", label="SF avec surventilation")
        plt.clim(min_, max_)
        plt.scatter(y3, z3, c=x3, marker="P", label="SF sans surventilation")
        plt.clim(min_, max_)
        plt.scatter(y2, z2, c=x2, marker="v", label="NoMASS")
        plt.clim(min_, max_)
        ax.set_ylabel("Cout actualisé en euros")
        ax.set_xlabel("Heures d'inconfort")
        #plt.legend(*plot.legend_elements("sizes", num=6))       
        fig.colorbar(plot,label="Besoins de chauffage kWh/(m2.an)")
        ax.legend()
    plt.show()
def plot_3d(df):
    x = df["f1"]#chauffage
    y = df["f2"] #inconfort
    z = df["f3"] #cout
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, alpha=0.5)
    plt.show()
if __name__=="__main__":
    #df_nomass_approche= pd.read_excel("./Results_To_Plot/nomass_approche.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    #df_all_gen_nomass= pd.read_excel("./Results_To_Plot/pareto_all_generations_nomass.xlsx", header=None, names=["x1","x2","x3","x4","f1","f2","f3"])
    #df_all_gen_deter= pd.read_excel("./Results_To_Plot/pareto_all_generations_deter.xlsx", header=None, names=["x1","x2","x3","x4","f1","f2","f3"])
    #df_avec_sur_all_comb=pd.read_excel("./Results_To_Plot/exhaustive_avec_surventilation_all_combinaisons.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_avec_sur_pareto=pd.read_excel("exhaustive_avec_surventilation_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_nomass_pareto=pd.read_excel("exhaustive_nomass_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_sans_sur_pareto=pd.read_excel("exhaustive_sans_surventilation_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_avec_sur_f1_f2_pareto=pd.read_excel("exhaustive_avec_sur_f1_f2_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_sans_sur_f1_f2_pareto=pd.read_excel("exhaustive_sans_sur_f1_f2_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_nomass_f1_f2_pareto=pd.read_excel("exhaustive_nomass_f1_f2_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_avec_sur_f1_f3_pareto=pd.read_excel("exhaustive_avec_sur_f1_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_sans_sur_f1_f3_pareto=pd.read_excel("exhaustive_sans_sur_f1_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_nomass_f1_f3_pareto=pd.read_excel("exhaustive_nomass_f1_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_avec_sur_f2_f3_pareto=pd.read_excel("exhaustive_avec_sur_f2_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_sans_sur_f2_f3_pareto=pd.read_excel("exhaustive_sans_sur_f2_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    df_nomass_f2_f3_pareto=pd.read_excel("exhaustive_nomass_f2_f3_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"]) 
    #df_sans_sur_pareto=pd.read_excel("exhaustive_sans_surventilation_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
    #df_deterministic_model=df_deterministic
    #df_nomass_model=df_nomass
    #df_nomass_approche_model=df_nomass_approche
    #df_deterministic_model["modèle"]="Scénarii fixes"
    #df_nomass_model["modèle"]="NoMASS"
    #df_nomass_approche_model["modèle"]="NoMASS Approché"
    #df_deter_nomass=pd.concat([df_deterministic_model,df_nomass_model])
    #df_deter_nomass_approche=pd.concat([df_deterministic_model,df_nomass_approche_model])
    #df_nomass_nomassApp=pd.concat([df_nomass_approche_model,df_nomass_model])
    #print(df_global)
    #sns.pairplot(df_nomass_nomassApp, hue='modèle',palette=["g", "r"],plot_kws={'alpha':0.5})
    #sns.pairplot(df_avec_sur_f1_f3_pareto)
    #plt.show()
    #sns.pairplot(df_nomass_approche,hue='x4',palette=["C0", "C1", "C2","C3"])
    #plot_pareto_f1_f3_SF(df_avec_sur_f2_f3_pareto,df_nomass_f2_f3_pareto,df_sans_sur_f2_f3_pareto,case=3)
    plot_density(df_avec_sur_pareto,df_nomass_pareto,df_sans_sur_pareto)
    #plot_3d(df_avec_sur_pareto)
    #plot_SRC(df_deterministic,df_nomass,df_nomass_approche)
    #all_generations()
    #plot_SRRC(df_deterministic,df_nomass,df_nomass_approche)
    #N_solutions_pareto_per_generation()
    