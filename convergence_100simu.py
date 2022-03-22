from matplotlib import style
import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np
import pandas as pd
def param(values):
    moyenne_cumulee=[]
    lower=[]
    upper=[]
    #print(statistics.mean(Hours[:3]))
    for i in range (len(values)):
        moyenne_cumulee.append(np.mean(values[:i+1]))
        alpha = 0.05                       # significance level = 5%
        df = len(values[:i+1]) - 1                  # degress of freedom = 20
        t = stats.t.ppf(1 - alpha/2, df)   # t-critical value for 95% CI = 2.093
        s = np.std(values[:i+1], ddof=1)            # sample standard deviation = 2.502
        n = len(values[:i+1])
        lower.append(np.mean(values[:i+1]) - (t * s / np.sqrt(n)))
        upper.append(np.mean(values[:i+1]) + (t * s / np.sqrt(n)))
    return moyenne_cumulee,lower,upper
def plot_hours(ax,values,values_occ):
    moyenne_cumulee,lower,upper=param(values)
    ax.plot(moyenne_cumulee,color='royalblue', label="Total")
    ax.plot(lower, linestyle=':',color='lightblue')
    ax.plot(upper, linestyle=':',color='lightblue')
    moyenne_cumulee_occ,lower_occ,upper_occ=param(values_occ)
    ax.plot(moyenne_cumulee_occ,color='green', label="Pendant l'occupation")
    ax.plot(lower_occ, linestyle=':',color='lightgreen',label="Intervalle de confiance à 95%")
    ax.plot(upper_occ, linestyle=':',color='lightgreen')
    return ax
def boxplot_hours(values_ann,values_ann_occ,values_ete,values_ete_occ):
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(1.9,3.8))
    c1='royalblue'
    c2='green'
    axes[0].boxplot(values_ann,showfliers=False,patch_artist=True,boxprops=dict(facecolor=c1, color=c1),capprops=dict(color=c1),
            whiskerprops=dict(color=c1),flierprops=dict(color=c1, markeredgecolor=c1))
    axes[0].boxplot([values_ann_occ],showfliers=False,patch_artist=True,boxprops=dict(facecolor=c2, color=c2),capprops=dict(color=c2),
            whiskerprops=dict(color=c2),flierprops=dict(color=c2, markeredgecolor=c2))
    axes[1].boxplot(values_ete,showfliers=False,patch_artist=True,boxprops=dict(facecolor=c1, color=c1),capprops=dict(color=c1),
            whiskerprops=dict(color=c1),flierprops=dict(color=c1, markeredgecolor=c1))
    axes[1].boxplot([values_ete_occ],showfliers=False,patch_artist=True,boxprops=dict(facecolor=c2, color=c2),capprops=dict(color=c2),
            whiskerprops=dict(color=c2),flierprops=dict(color=c2, markeredgecolor=c2))
    axes[0].set_ylim(0,5000)
    plt.savefig("boxplot_hours.png")
    plt.show()
def plot_hours_convergence(annee,annee_occ,ete,ete_occ):
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(8,3.8))
    ax0=plot_hours(axes[0],annee,annee_occ)
    ax0.set_ylabel("Nombre d'heures d'inconfort")
    ax0.set_xlabel("Nombre de simulations")
    ax0.set_title("Annuel")
    ax0.legend(fontsize=7)
    ax1=plot_hours(axes[1],ete,ete_occ)
    ax1.set_xlabel("Nombre de simulations")
    ax1.set_title("Estival")
    ax1.legend(fontsize=7)
    ax0.set_ylim(0,5000)
    ax0.grid(axis = 'y')
    ax1.grid(axis = 'y')
    plt.savefig("convergence_Hours")
    plt.show()
def plot_DH_convergence(annee,annee_occ,ete,ete_occ):
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(8,3.8))
    ax0=plot_hours(axes[0],annee,annee_occ)
    ax0.set_ylabel("DH [°C.h]")
    ax0.set_xlabel("Nombre de simulations")
    ax0.set_title("Annuel")
    ax0.legend(fontsize=7)
    ax1=plot_hours(axes[1],ete,ete_occ)
    ax1.set_xlabel("Nombre de simulations")
    ax1.set_title("Estival")
    ax1.legend(fontsize=7)
    ax0.set_ylim(0,25000)
    ax0.grid(axis = 'y')
    ax1.grid(axis = 'y')
    #ax0.axhline(y = 350, color = 'r', linestyle = '--')
    #ax1.axhline(y = 350, color = 'r', linestyle = '--')
    plt.savefig("convergence_DH")
    plt.show()
annee= pd.read_excel("./Results_To_Plot/overheating_annee_sans.xlsx", names=["DH","H"])
annee_occ= pd.read_excel("./Results_To_Plot/overheating_annee_occ.xlsx", names=["DH","H"])
ete= pd.read_excel("./Results_To_Plot/overheating_ete_sans.xlsx", names=["DH","H"])
ete_occ= pd.read_excel("./Results_To_Plot/overheating_ete_occ.xlsx", names=["DH","H"])
DH_annee=annee["DH"].values.tolist()
DH_annee_occ=annee_occ["DH"].values.tolist()
DH_ete=ete["DH"].values.tolist()
DH_ete_occ=ete_occ["DH"].values.tolist()
Hours_annee=annee["H"].values.tolist()
Hours_annee_occ=annee_occ["H"].values.tolist()
Hours_ete=ete["H"].values.tolist()
Hours_ete_occ=ete_occ["H"].values.tolist()

#boxplot_hours(Hours_annee,Hours_annee_occ,Hours_ete,Hours_ete_occ)
plot_hours_convergence(Hours_annee,Hours_annee_occ,Hours_ete,Hours_ete_occ)
plot_DH_convergence(DH_annee,DH_annee_occ,DH_ete,DH_ete_occ)