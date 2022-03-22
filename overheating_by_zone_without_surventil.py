import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend import Legend
def plot_DH():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,4))
    plt.subplots_adjust(wspace=0.1,left=0.25,top=0.91,bottom=0.3)
    labels=["SF","NoMASS_GM","NoMASS_GM_F","NoMASS"]
    ann_total=[[21729,16521,7661,5946],#RDC
        [26289,19263,9298,6899],#CH1 
        [22630,17142,7334,5382],#CH2
        [22688,16420,4946,3818],#CH3
        [24398,20314,11510,9424]]#SDB
    ann_occ=[[10591,11924,6107,4872],#RDC
        [9410,7909,3342,2411],#CH1 
        [7485,7110,2553,1782],#CH2
        [7404,613,184,147],#CH3
        [1190,4127,2532,2142]]#SDB
    ete_total=[[15887,12550,5315,4371],#RDC
        [18624,14392.8,6421,5185],#CH1 
        [16500,12999,5099,4074],#CH2
        [16516,12674,3240,2728],#CH3
        [17586,14866,7923,6884]]#SDB
    ete_occ=[[7218,8761,4099,3484],#RDC
        [6658,6024,2356,1870],#CH1 
        [5592,5547,1840,1407],#CH2
        [5543,449,107,95],#CH3
        [824,2919,1670,1493]]#SDB
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    n_rows = len(ann_total)
    barWidth = 0.25
    br1 = np.arange(len(labels))
    
    # Initialize the vertical-offset for the stacked bar chart.
    y_ann_total = np.zeros(len(labels))
    y_ann_occ = np.zeros(len(labels))
    y_ete_total = np.zeros(len(labels))
    y_ete_occ = np.zeros(len(labels))
    # Plot bars and create text labels for the table
    for row in range(n_rows):
        axes[0].bar(br1-barWidth/2, ann_total[row], barWidth, bottom=y_ann_total, color=colors[row], label=rows[row])
        axes[0].bar(br1+barWidth/2, ann_occ[row], barWidth, bottom=y_ann_occ, color=colors[row],hatch="..")
        y_ann_total = y_ann_total + ann_total[row]
        y_ann_occ = y_ann_occ + ann_occ[row]
        axes[1].bar(br1-barWidth/2, ete_total[row], barWidth, bottom=y_ete_total, color=colors[row], label=rows[row])
        axes[1].bar(br1+barWidth/2, ete_occ[row], barWidth, bottom=y_ete_occ, color=colors[row],hatch="..")
        y_ete_total = y_ete_total + ete_total[row]
        y_ete_occ = y_ete_occ + ete_occ[row]
    #axes[1].axhline(y = 350, color = 'r', linestyle = '-')
    axes[0].set_ylabel('DH (°C.h)')
    axes[0].set_title('Année (hors_chauffe)')
    #axes[0].set_xticklabels(labels,fontsize=7)
    axes[1].set_title('Eté')
    #axes[1].set_xticklabels(labels,rotation=45)
    #plt.xticks(br1,labels)
    axes[0].set_xticks(br1) 
    axes[0].set_xticklabels(labels)
    axes[0].tick_params(axis="x",rotation=45) 
    axes[1].tick_params(axis="x",rotation=45)
    #plt.setp(axes,xticks=br1,xticklabels=labels)
    #axes[0].set_xticklabels(fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone')
    axes[1].grid(axis = 'y')
    #axes[0].set_ylim(0,25000)
    axes[0].grid(axis = 'y')
    plt.savefig("DH_by_zone.png")
    plt.show()
def plot_Hours():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,4))
    plt.subplots_adjust(wspace=0.1,left=0.25,top=0.91,bottom=0.3)
    labels=["SF","NoMASS_GM","NoMASS_GM_F","NoMASS"]
    ann_total=[[4081.5,3840,3285,2841],#RDC
        [4371,4178,3682,3109],#CH1 
        [4308,4032,3428,2903],#CH2
        [4313,3877,2354,2020],#CH3
        [4352,4254,4046,3557]]#SDB
    ann_occ=[[1649,2576,2319,2065],#RDC
        [1566,1840,1554,1276],#CH1 
        [1534,1821,1453,1182],#CH2
        [1532,135,84,75],#CH3
        [183,792,771,715]]#SDB
    ete_total=[[2256,2255,2013,1889],#RDC
        [2256,2256,2056,1986],#CH1 
        [2256,2256,2006,1907],#CH2
        [2256,2256,1271,1224],#CH3
        [2256,2256,2245,2203]]#SDB
    ete_occ=[[840,1427,1330,1281],#RDC
        [804,1001,876,841],#CH1 
        [804,1042,883,824],#CH2
        [804,72,40,40],#CH3
        [94,415,414,410]]#SDB
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    n_rows = len(ann_total)
    barWidth = 0.25
    br1 = np.arange(len(labels))
    # Initialize the vertical-offset for the stacked bar chart.
    y_ann_total = np.zeros(len(labels))
    y_ann_occ = np.zeros(len(labels))
    y_ete_total = np.zeros(len(labels))
    y_ete_occ = np.zeros(len(labels))
    # Plot bars and create text labels for the table
    for row in range(n_rows):
        axes[0].bar(br1-barWidth/2, ann_total[row], barWidth, bottom=y_ann_total, color=colors[row], label=rows[row])
        axes[0].bar(br1+barWidth/2, ann_occ[row], barWidth, bottom=y_ann_occ, color=colors[row],hatch="..")
        y_ann_total = y_ann_total + ann_total[row]
        y_ann_occ = y_ann_occ + ann_occ[row]
        axes[1].bar(br1-barWidth/2, ete_total[row], barWidth, bottom=y_ete_total, color=colors[row], label=rows[row])
        axes[1].bar(br1+barWidth/2, ete_occ[row], barWidth, bottom=y_ete_occ, color=colors[row],hatch="..")
        y_ete_total = y_ete_total + ete_total[row]
        y_ete_occ = y_ete_occ + ete_occ[row]
    #axes[1].axhline(y = 350, color = 'r', linestyle = '-')
    axes[0].set_ylabel("Nombre d'heures d'inconfort")
    axes[0].set_title('Année (hors_chauffe)')
    axes[1].set_title('Eté')
    plt.xticks(br1,labels)
    axes[0].tick_params(axis="x",rotation=45) 
    axes[1].tick_params(axis="x",rotation=45)
    '''axes[0].set_xticklabels(labels,rotation=45)
    axes[1].set_xticklabels(labels,rotation=45)'''
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles[::-1], labels[::-1], title='Zone')
    axes[1].grid(axis = 'y')
    axes[0].grid(axis = 'y')
    plt.savefig("Hours_by_zone.png")
    plt.show()
def plot_FEN():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(4,3.5))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["NoMASS"]
    ann_total=[[320.6],#RDC
        [159],#CH1 
        [159.4],#CH2
        [999]]#CH3
    ann_occ=[[221.5],#RDC
        [90],#CH1 
        [84],#CH2
        [32]]#CH3
    ete_total=[[250.6],#RDC
        [122],#CH1 
        [126],#CH2
        [757]]#CH3
    ete_occ=[[173.75],#RDC
        [70],#CH1 
        [68.4],#CH2
        [24]]#CH3
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    n_rows = len(ann_total)
    barWidth = 0.2
    br1 = np.arange(len(labels))
    # Initialize the vertical-offset for the stacked bar chart.
    y_ann_total = np.zeros(len(labels))
    y_ann_occ = np.zeros(len(labels))
    y_ete_total = np.zeros(len(labels))
    y_ete_occ = np.zeros(len(labels))
    # Plot bars and create text labels for the table
    for row in range(n_rows):
        axes[0].bar(br1-barWidth/2, ann_total[row], barWidth, bottom=y_ann_total, color=colors[row], label=rows[row])
        axes[0].bar(br1+barWidth/2, ann_occ[row], barWidth, bottom=y_ann_occ, color=colors[row],hatch="..")
        y_ann_total = y_ann_total + ann_total[row]
        y_ann_occ = y_ann_occ + ann_occ[row]
        axes[1].bar(br1-barWidth/2, ete_total[row], barWidth, bottom=y_ete_total, color=colors[row], label=rows[row])
        axes[1].bar(br1+barWidth/2, ete_occ[row], barWidth, bottom=y_ete_occ, color=colors[row],hatch="..")
        y_ete_total = y_ete_total + ete_total[row]
        y_ete_occ = y_ete_occ + ete_occ[row]
    axes[0].set_ylabel("Nombre d'heures d'ouverture des fenêtres")
    plt.xticks(br1,labels)
    first_legend=mpatches.Patch(color='r',hatch='..',label="Pendant l'occupation")#
    '''axes[0].add_artist(first_legend)'''
    handles, labels = axes[0].get_legend_handles_labels()
    #handles.append(first_legend)
    fig.legend(handles[::-1], labels[::-1], title='Zone',fontsize=8)
    #axes[0].add_artist(first_legend)
    axes[0].set_title('Année (hors_chauffe)',fontsize=10)
    axes[1].set_title('Eté',fontsize=10)
    axes[1].grid(axis = 'y')
    axes[0].grid(axis = 'y')
    
    #plt.savefig("FEN_by_zone.png")
    plt.show()
def plot_Hours_per_occ():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,3.8))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["SF_sans","SF_avec","NoMASS"]
    ann_occ=np.array([[1649,1406,2065],#RDC
        [1566,405,1276],#CH1 
        [1534,348,1182],#CH2
        [1532,346,75],#CH3
        [183,182,715]])#SDB
    hours_ann=np.array([[1652,1652,2781],#RDC
        [1570,1570,1951],#CH1 
        [1570,1570,2034],#CH2
        [1570,1570,142],#CH3
        [183,183,801]])#SDB
    hours_ann_percentage=ann_occ/hours_ann*100
    ete_occ=np.array([[840,698,1281],#RDC
        [804,62,841],#CH1 
        [804,65,824],#CH2
        [804,66.5,40],#CH3
        [94,94,410]])#SDB
    hours_ete=np.array([[840,840,1427],#RDC
        [804,804,1001],#CH1 
        [804,804,1042],#CH2
        [804,804,72],#CH3
        [94,94,415]])#SDB
    hours_ete_percentage=ete_occ/hours_ete*100
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    n_rows = len(hours_ann)
    barWidth = 0.25
    br1 = np.arange(len(labels))
    y_hours_ann = np.zeros(len(labels))
    y_hours_ete = np.zeros(len(labels))
    for row in range(n_rows):
        axes[0].bar(br1+barWidth/2, hours_ann_percentage[row], barWidth, bottom=y_hours_ann, color=colors[row],hatch="..", label=rows[row])
        y_hours_ann = y_hours_ann + hours_ann_percentage[row]
        axes[1].bar(br1+barWidth/2, hours_ete_percentage[row], barWidth, bottom=y_hours_ete, color=colors[row],hatch="..", label=rows[row])
        y_hours_ete = y_hours_ete + hours_ete_percentage[row]
    axes[0].set_ylabel("Pourcentage inconfort pendant occupation")
    axes[0].set_title('Année (hors_chauffe)')
    axes[1].set_title('Eté')
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles[::-1], labels[::-1], title='Zone')
    axes[1].grid(axis = 'y')
    axes[0].grid(axis = 'y')
    plt.savefig("Hours_occupation_by_zone.png")
    plt.show()
def plot_Heures_occ():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(4,3.8))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["SF","NoMASS"]
    hours_ann=np.array([[1652,2781],#RDC
        [1570,1951],#CH1 
        [1570,2034],#CH2
        [1570,142],#CH3
        [183,801]])#SDB
    hours_ete=np.array([[840,1427],#RDC
        [804,1001],#CH1 
        [804,1042],#CH2
        [804,72],#CH3
        [94,415]])#SDB
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = ['royalblue','darkgreen','darkorange','orchid','y']#plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
    n_rows = len(hours_ann)
    barWidth = 0.5
    br1 = np.arange(len(labels))
    y_hours_ann = np.zeros(len(labels))
    y_hours_ete = np.zeros(len(labels))
    for row in range(n_rows):
        axes[0].bar(br1, hours_ann[row], barWidth, bottom=y_hours_ann, color=colors[row],hatch="..", label=rows[row])
        y_hours_ann = y_hours_ann + hours_ann[row]
        axes[1].bar(br1, hours_ete[row], barWidth, bottom=y_hours_ete, color=colors[row],hatch="..", label=rows[row])
        y_hours_ete = y_hours_ete + hours_ete[row]
    axes[0].set_ylabel("Nombre d'heures d'occupation")
    axes[0].set_title('Année (hors_chauffe)',fontsize=10)
    axes[1].set_title('Eté',fontsize=10)
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone',fontsize=10)
    axes[1].grid(axis = 'y')
    axes[0].grid(axis = 'y')
    plt.savefig("Heures_occupation_by_zone.png")
    plt.show()
plot_DH()
plot_Hours()
#plot_FEN()
#plot_Heures_occ()