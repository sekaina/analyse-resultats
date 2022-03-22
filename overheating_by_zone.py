import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_DH():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,3.8))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["SF_sans","SF_avec","NoMASS"]
    ann_total=[[21729,5120,5946],#RDC
        [26289,2735,6899],#CH1 
        [22630,2335,5382],#CH2
        [22688,2643,3818],#CH3
        [24398,7726,9424]]#SDB
    ann_occ=[[10591,3696,4872],#RDC
        [9410,326,2411],#CH1 
        [7485,166,1782],#CH2
        [7404,169,147],#CH3
        [1190,481,2142]]#SDB
    ete_total=[[15887,2910,4371],#RDC
        [18624,1180,5185],#CH1 
        [16500,968,4074],#CH2
        [16516,1147,2728],#CH3
        [17586,4687,6884]]#SDB
    ete_occ=[[7218,2049,3484],#RDC
        [6658,18,1870],#CH1 
        [5592,15,1407],#CH2
        [5543,15,95],#CH3
        [824,277,1493]]#SDB
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
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
    axes[0].set_title('Annuel (hors_chauffe)')
    axes[1].set_title('Estival')
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone')
    '''axes[1].grid(axis = 'y')
    axes[0].set_ylim(0,25000)
    axes[0].grid(axis = 'y')'''
    plt.savefig("DH_by_zone.png")
    plt.show()
def plot_Hours():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,3.8))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["SF_sans","SF_avec","NoMASS"]
    ann_total=[[4081.5,2468,2841],#RDC
        [4371,2260.5,3109],#CH1 
        [4308,2246,2903],#CH2
        [4313,2342,2020],#CH3
        [4352,4192,3557]]#SDB
    ann_occ=[[1649,1406,2065],#RDC
        [1566,405,1276],#CH1 
        [1534,348,1182],#CH2
        [1532,346,75],#CH3
        [183,182,715]]#SDB
    ete_total=[[2256,1290.5,1889],#RDC
        [2256,933,1986],#CH1 
        [2256,980,1907],#CH2
        [2256,1039,1224],#CH3
        [2256,2249,2203]]#SDB
    ete_occ=[[840,698,1281],#RDC
        [804,62,841],#CH1 
        [804,65,824],#CH2
        [804,66.5,40],#CH3
        [94,94,410]]#SDB
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
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
    axes[0].set_title('Annuel (hors_chauffe)')
    axes[1].set_title('Estival')
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone')
    '''axes[1].grid(axis = 'y')
    axes[0].set_ylim(0,25000)
    axes[0].grid(axis = 'y')'''
    plt.savefig("Hours_by_zone.png")
    plt.show()
def plot_FEN():
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(5,3.8))
    plt.subplots_adjust(wspace=0.1,left=0.18)
    labels=["SF_avec","NoMASS"]
    ann_total=[[831,320.6],#RDC
        [451.3,159],#CH1 
        [298.6,159.4],#CH2
        [311,999]]#CH3
    ann_occ=[[275.4,221.5],#RDC
        [301,90],#CH1 
        [180,84],#CH2
        [186.3,32]]#CH3
    ete_total=[[651.7,250.6],#RDC
        [355,122],#CH1 
        [242.8,126],#CH2
        [251,757]]#CH3
    ete_occ=[[174.75,173.75],#RDC
        [241,70],#CH1 
        [150.5,68.4],#CH2
        [154,24]]#CH3
    rows=['RDC','CH1','CH2','CH3','SDB']
    colors = plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
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
    axes[0].set_ylabel("Nombre d'heures d'ouverture des fenêtres")
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone')
    axes[0].set_title('Annuel (hors_chauffe)')
    axes[1].set_title('Estival')
    plt.savefig("FEN_by_zone.png")
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
    colors = plt.cm.BuPu(np.linspace(0.5, 1, len(rows)))
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
    axes[0].set_title('Annuel (hors_chauffe)')
    axes[1].set_title('Estival')
    plt.xticks(br1,labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title='Zone')
    plt.savefig("Hours_occupation_by_zone.png")
    plt.show()
#plot_DH()
#plot_Hours()
#plot_FEN()
plot_Hours_per_occ()