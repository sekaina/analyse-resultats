from matplotlib import style
import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np
import pandas as pd
def param(values):
    moyenne_cumulee=[]
    lower=[]
    upper=[]
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
def plot_hours(ax,values,values_occ,L,C):
    moyenne_cumulee,lower,upper=param(values)
    ax.plot(moyenne_cumulee, label=L+'_Tot', color=C)
    ax.plot(lower, linestyle=':',color=C,linewidth=0.5)
    ax.plot(upper, linestyle=':',color=C,linewidth=0.5)
    moyenne_cumulee_occ,lower_occ,upper_occ=param(values_occ)
    ax.plot(moyenne_cumulee_occ,marker='o', label=L+'_Occ', color=C,markersize=2)
    ax.plot(lower_occ, linestyle=':',color=C,linewidth=0.5)
    ax.plot(upper_occ, linestyle=':',color=C,linewidth=0.5)
    return ax
def plot_hours_convergence(annee,annee_occ,ete,ete_occ):
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(8,3.8))
    plt.subplots_adjust(wspace=0.05,left=0.1,right=0.88,top=0.75,bottom=0.13)
    #marqueurs = {'RDC':'o', 'CH1':'s', 'CH2':'x', 'CH3':'*', 'SDB':'^'}
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    for row in range(len(rows)):
        ax0=plot_hours(axes[0],annee[rows[row]],annee_occ[rows[row]],rows[row],colors[row])
        ax1=plot_hours(axes[1],ete[rows[row]],ete_occ[rows[row]],rows[row],colors[row])
    ax0.set_ylabel("Nombre d'heures d'inconfort")
    ax0.set_xlabel("Nombre de simulations")
    ax0.set_title("Année")
    fig.legend(fontsize=7,loc='upper right')  
    ax1.set_xlabel("Nombre de simulations")
    ax1.set_title("Eté")
    #ax0.set_ylim(0,5000)
    ax0.grid(axis = 'y')
    ax1.grid(axis = 'y')
    plt.savefig("convergence_Hours_by_zone.png")
    plt.show()
def plot_DH_convergence(annee,annee_occ,ete,ete_occ):
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(8,3.8))
    plt.subplots_adjust(wspace=0.05,left=0.1,right=0.88,top=0.75,bottom=0.13)
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']
    for row in range(len(rows)):
        ax0=plot_hours(axes[0],annee[rows[row]],annee_occ[rows[row]],rows[row],colors[row])
        ax1=plot_hours(axes[1],ete[rows[row]],ete_occ[rows[row]],rows[row],colors[row])
    ax0.set_ylabel("DH [°C.h]")
    ax0.set_xlabel("Nombre de simulations")
    ax0.set_title("Année")
    fig.legend(fontsize=7,loc='upper right')
    ax1.set_xlabel("Nombre de simulations")
    ax1.set_title("Eté")
    ax0.set_ylim(0,10000)
    ax0.grid(axis = 'y')
    ax1.grid(axis = 'y')
    #ax0.axhline(y = 350, color = 'r', linestyle = '--')
    #ax1.axhline(y = 350, color = 'r', linestyle = '--')
    plt.savefig("convergence_DH_by_zone")
    plt.show()

Hours_ete_total= pd.read_csv("./Hours_conv_ete_total.csv")
Hours_ann_total= pd.read_csv("./Hours_conv_ann_total.csv")
Hours_ete_occ= pd.read_csv("./Hours_conv_ete_occ.csv")
Hours_ann_occ= pd.read_csv("./Hours_conv_ann_occ.csv")
DH_ete_total= pd.read_csv("./DH_conv_ete_total.csv")
DH_ann_total= pd.read_csv("./DH_conv_ann_total.csv")
DH_ete_occ= pd.read_csv("./DH_conv_ete_occ.csv")
DH_ann_occ= pd.read_csv("./DH_conv_ann_occ.csv")

#DH_annee=annee["DH"].values.tolist()


#boxplot_hours(Hours_annee,Hours_annee_occ,Hours_ete,Hours_ete_occ)
#plot_hours_convergence(Hours_ann_total,Hours_ann_occ,Hours_ete_total,Hours_ete_occ)
plot_DH_convergence(DH_ann_total,DH_ann_occ,DH_ete_total,DH_ete_occ)