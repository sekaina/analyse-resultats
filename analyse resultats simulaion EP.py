import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
zones=["RDCTHERMALZONE","ZCH1","ZCH2","ZCH3","ZSDB"]
variables=["WINDOWSTATE0:Schedule Value [](TimeStep)","BLINDFRACTION:Schedule Value [](TimeStep)", "NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"]
df_nomass1=pd.read_csv("./Results_to_Plot/IDM_NoMASS.csv")#, index_col="Date/Time")
df_nomass2=pd.read_csv("./Results_to_Plot/IDM_NoMASS_seed1000.csv")
df_deter=pd.read_csv("./Results_to_Plot/IDM.csv")#, index_col="Date/Time")
df_sans_surv=pd.read_csv("./Results_to_Plot/IDM_sans_surventilation.csv")
def taux_occupation_nomass(df):
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2","CH3","RDC"])
    '''ch1=df["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].mean()
    ch2=df["ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].mean()
    ch3=df["ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].mean()
    rdc=df["RDCTHERMALZONENUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].mean()
    sdb=df["ZSDBNUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].mean()'''
    occup=df.iloc[:, ["NUMBEROFOCCUPANTS" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in occup.columns:
        volet_moy=[float(occup[col][i+84:i+252].mean()) for i in range(0,len(occup),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
        if "SDB" in col:
            df_result["SDB"]=volet_moy
    return df_result
    #return ch1+ch2+ch3+rdc+sdb
def plot_occupation(df):
    taux_occupation_nomass(df).sum(axis=1).plot()
    plt.show()
def taux_volet_nomass_jour(df):
    blind=df.iloc[:, ["BLINDFRACTION" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in blind.columns:
        volet_moy=[float(blind[col][i+84:252+i].mean()) for i in range(0,len(blind),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
    return df_result
def taux_fen_nomass_jour(df):
    blind=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in blind.columns:
        volet_moy=[float(blind[col][i+84:252+i].mean()) for i in range(0,len(blind),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
    return df_result
def taux_fen_nomass_nuit(df):
    blind=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in blind.columns:
        volet_moy=[float(blind[col][i+252:+i+372].mean()) for i in range(0,len(blind),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
    return df_result
def plot_taux_fen(df,df2):
    fig,axes=plt.subplots(nrows=4,ncols=2,sharex=True, sharey=True)
    jour=taux_fen_nomass_jour(df)
    jour2=taux_fen_nomass_jour(df2)
    axes[0,0].plot(jour["RDC"], label="RDC (seed 0)")
    axes[1,0].plot(jour["CH1"], label="CH1 (seed 0)")
    axes[2,0].plot(jour["CH2"], label="CH2 (seed 0)")
    axes[3,0].plot(jour["CH3"], label="CH3 (seed 0)")

    axes[0,0].plot(jour2["RDC"], label="RDC (seed 1000)")
    axes[1,0].plot(jour2["CH1"], label="CH1 (seed 1000)")
    axes[2,0].plot(jour2["CH2"], label="CH2 (seed 1000)")
    axes[3,0].plot(jour2["CH3"], label="CH3 (seed 1000)")
    nuit=taux_fen_nomass_nuit(df)
    nuit2=taux_fen_nomass_nuit(df2)
    axes[0,1].plot(nuit["RDC"], label="RDC (seed 0)")
    axes[1,1].plot(nuit["CH1"], label="CH1 (seed 0)")
    axes[2,1].plot(nuit["CH2"], label="CH2 (seed 0)")
    axes[3,1].plot(nuit["CH3"], label="CH3 (seed 0)")

    axes[0,1].plot(nuit2["RDC"], label="RDC (seed 1000)")
    axes[1,1].plot(nuit2["CH1"], label="CH1 (seed 1000)")
    axes[2,1].plot(nuit2["CH2"], label="CH2 (seed 1000)")
    axes[3,1].plot(nuit2["CH3"], label="CH3 (seed 1000)")
    axes[0,0].set_title("taux d'ouverture des fenêtres entre 7h et 21h")
    axes[0,1].set_title("taux d'ouverture des fenêtres entre 21h et 7h")
    axes[0,1].legend(fontsize=7)
    axes[1,1].legend(fontsize=7)
    axes[2,1].legend(fontsize=7)
    axes[3,1].legend(fontsize=7)
    plt.show()
def heure_fen_ouvert_jour(df):
    """durée d'ouverture en nombre d'heures des fenetres entre 7h et 21h pour chaque jour"""
    fen=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2","CH3"])
    for col in fen.columns:
        if "RDC" in col:
            df_result["RDC"]=[float(fen[col][i+84:252+i][fen[col]==1].count()) for i in range(0,len(fen),288)]
        if "CH1" in col:
            df_result["CH1"]=[float(fen[col][i+84:252+i][fen[col]==1].count()) for i in range(0,len(fen),288)]
        if "CH2" in col:
            df_result["CH2"]=[float(fen[col][i+84:252+i][fen[col]==1].count()) for i in range(0,len(fen),288)]
        if "CH3" in col:
            df_result["CH3"]=[float(fen[col][i+84:252+i][fen[col]==1].count()) for i in range(0,len(fen),288)]
    return df_result*5/60
def heure_fen_ouvert_jour_annee(df):
    """durée d'ouverture en nombre d'heures des fenetres entre 7h et 21h pour chaque jour"""
    fen=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2","CH3"])
    for col in fen.columns:
        if "RDC" in col:
            df_result["RDC"]=[fen[col][i+84:252+i].sum() for i in range(0,len(fen),288)]
        if "CH1" in col:
            df_result["CH1"]=[fen[col][i+84:252+i].sum() for i in range(0,len(fen),288)]
        if "CH2" in col:
            df_result["CH2"]=[fen[col][i+84:252+i].sum() for i in range(0,len(fen),288)]
        if "CH3" in col:
            df_result["CH3"]=[fen[col][i+84:252+i].sum() for i in range(0,len(fen),288)]
        if "SDB" in col:
            df_result["SDB"]=[fen[col][i+84:252+i].sum() for i in range(0,len(fen),288)]
    return df_result.sum()*5/60
def heure_fen_ouvert_nuit_annee(df):
    """durée d'ouverture en nombre d'heures des fenetres entre 7h et 21h pour chaque jour"""
    fen=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2","CH3"])
    for col in fen.columns:
        if "RDC" in col:
            df_result["RDC"]=[fen[col][i+252:372+i].sum() for i in range(0,len(fen),288)]
        if "CH1" in col:
            df_result["CH1"]=[fen[col][i+252:372+i].sum() for i in range(0,len(fen),288)]
        if "CH2" in col:
            df_result["CH2"]=[fen[col][i+252:372+i].sum() for i in range(0,len(fen),288)]
        if "CH3" in col:
            df_result["CH3"]=[fen[col][i+252:372+i].sum() for i in range(0,len(fen),288)]
        if "SDB" in col:
            df_result["SDB"]=[fen[col][i+252:372+i].sum() for i in range(0,len(fen),288)]
    return df_result.sum()*5/60
def heure_fen_ouvert_annee(df):
    """durée d'ouverture en nombre d'heures des fenetres annuel"""
    fen=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=fen.sum()
    return df_result*5/60
def heure_fen_ouvert_hors_chauff(df):
    """durée d'ouverture en nombre d'heures des fenetres annuel"""
    fen=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    df_result=fen[29954:82370].sum()
    return df_result*5/60
def taux_eclairage_nomass_jour(df):
    blind=df.iloc[:, ["LIGHTSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in blind.columns:
        volet_moy=[float(blind[col][i+84:252+i].mean()) for i in range(0,len(blind),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
    return df_result
def taux_eclairage_nomass_nuit(df):
    blind=df.iloc[:, ["LIGHTSTATE" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2"])
    for col in blind.columns:
        volet_moy=[float(blind[col][i+252:+i+372].mean()) for i in range(0,len(blind),288)]
        if "RDC" in col:
            df_result["RDC"]=volet_moy
        if "CH1" in col:
            df_result["CH1"]=volet_moy
        if "CH2" in col:
            df_result["CH2"]=volet_moy
        if "CH3" in col:
            df_result["CH3"]=volet_moy
    return df_result
def plot_eclairage_volet(df,df2):
    fig,axes=plt.subplots(nrows=4,ncols=2,sharex=True, sharey=True)
    eclairage1=taux_eclairage_nomass_jour(df)
    volet1=taux_volet_nomass_jour(df)
    eclairage2=taux_eclairage_nomass_jour(df2)
    volet2=taux_volet_nomass_jour(df2)
    axes[0,0].plot(eclairage1["RDC"], label="RDC (éclairage)")
    axes[1,0].plot(eclairage1["CH1"], label="CH1 (éclairage)")
    axes[2,0].plot(eclairage1["CH2"], label="CH2 (éclairage)")
    axes[3,0].plot(eclairage1["CH3"], label="CH3 (éclairage)")

    axes[0,0].plot(volet1["RDC"], label="RDC (volets)")
    axes[1,0].plot(volet1["CH1"], label="CH1 (volets)")
    axes[2,0].plot(volet1["CH2"], label="CH2 (volets)")
    axes[3,0].plot(volet1["CH3"], label="CH3 (volets)")

    axes[0,1].plot(eclairage2["RDC"], label="RDC (éclairage)")
    axes[1,1].plot(eclairage2["CH1"], label="CH1 (éclairage)")
    axes[2,1].plot(eclairage2["CH2"], label="CH2 (éclairage)")
    axes[3,1].plot(eclairage2["CH3"], label="CH3 (éclairage)")

    axes[0,1].plot(volet2["RDC"], label="RDC (volets)")
    axes[1,1].plot(volet2["CH1"], label="CH1 (volets)")
    axes[2,1].plot(volet2["CH2"], label="CH2 (volets)")
    axes[3,1].plot(volet2["CH3"], label="CH3 (volets)")

    axes[0,0].set_title("seed 0")
    axes[0,1].set_title("seed 1000")
    axes[0,1].legend(fontsize=7)
    axes[1,1].legend(fontsize=7)
    axes[2,1].legend(fontsize=7)
    axes[3,1].legend(fontsize=7)
    plt.show()
def temp_max(df):
    temp=df.iloc[:, ["Zone Mean Air Temperature [C](TimeStep)" in col for col in df.columns]]
    return temp.max()
def plot_ch1_2502(df):
    fig,axes=plt.subplots(nrows=3,ncols=1,sharex=True, figsize=(5,5))
    ax0=axes[0].twinx()
    ax1=axes[1].twinx()
    df.iloc[15843:15843+288].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0],c='b',label="ZCH1: Temperature Intérieure[C]").legend(fontsize=7)
    df.iloc[15843:15843+288].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0],c='g',label="Temperature Extérieure [C]").legend(fontsize=7)
    df.iloc[15843:15843+288].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(fontsize=7)
    df.iloc[15843:15843+288].plot("Date/Time","ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1],yticks=[0,1],c='b',label="ZCH1: fenêtre").legend(fontsize=7)
    df.iloc[15843:15843+288].plot("Date/Time","ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1,c='r',style=':',label="ZCH1: débit de renouvellement d'air [Vol/h]").legend(fontsize=7)
    df.iloc[15843:15843+288].plot("Date/Time","ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[2],c='b',label="ZCH1: Nombre Occupants").legend(fontsize=7)
    axes[0].locator_params(axis='x',nbins=8)
    #axes[0].set_xticks(fontsize=7)
    plt.savefig(".\graphes\ch1_2502.png")
    plt.show()
def plot_hist_duree(d1,d2):
    fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True)
    b=[0,60,120,300,480,720,1440,2880,4320,5760]
    #b=[i*5 for i in b]
    axes[0,0].hist([d1["CH1"],d2["CH1"]],bins=b,label=["seed 0","seed 1000"])
    axes[1,0].hist([d1["CH3"],d2["CH3"]],bins=b)
    axes[0,1].hist([d1["CH2"],d2["CH2"]],bins=b)
    axes[1,1].hist([d1["RDC"],d2["RDC"]],bins=b)
    #t=[0,120,240,360,480,600,720,840,960,1080]
    t=[0,60,120,300,480,720,1440,2880,4320,5760]
    #t=[i*5 for i in t]
    plt.xticks(ticks=t)
    fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True)  
    fig.legend()
    axes[0,0].set_title("CH1")
    axes[1,0].set_title("CH3")
    axes[0,1].set_title("CH2")
    axes[1,1].set_title("RDC")
    plt.show()
def plot_hist_temp(df_sans_surv,df_deter,df_nomass1, df_nomass2):
    fig,axes=plt.subplots(nrows=2,ncols=2,sharey=True)
    m=df_sans_surv.shape[0]
    n00,x00,_=axes[0,0].hist([df_sans_surv["ZCH1:Zone Mean Air Temperature [C](TimeStep)"],df_deter["ZCH1:Zone Mean Air Temperature [C](TimeStep)"],
            df_nomass1["ZCH1:Zone Mean Air Temperature [C](TimeStep)"],df_nomass2["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]], 
            label=["Scénariii fixes sans surventilation nocturne","Scénarii fixes avec surventilation nocturne","NoMASS (seed 0)","NoMASS (seed 1000)"],
            bins=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    n10,x10,_=axes[1,0].hist([df_sans_surv["ZCH3:Zone Mean Air Temperature [C](TimeStep)"],df_deter["ZCH3:Zone Mean Air Temperature [C](TimeStep)"],
            df_nomass1["ZCH3:Zone Mean Air Temperature [C](TimeStep)"],df_nomass2["ZCH3:Zone Mean Air Temperature [C](TimeStep)"]], 
            bins=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    n01,x01,_=axes[0,1].hist([df_sans_surv["ZCH2:Zone Mean Air Temperature [C](TimeStep)"],df_deter["ZCH2:Zone Mean Air Temperature [C](TimeStep)"],
            df_nomass1["ZCH2:Zone Mean Air Temperature [C](TimeStep)"],df_nomass2["ZCH2:Zone Mean Air Temperature [C](TimeStep)"]], 
            bins=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    n11,x11,_=axes[1,1].hist([df_sans_surv["RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)"],df_deter["RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)"],
            df_nomass1["RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)"],df_nomass2["RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)"]], 
            bins=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    
    fig.legend()
    fig,axes=plt.subplots(nrows=2,ncols=2,sharey=True,sharex=True)
    t=[14,16,18,20,22,24,26,28,30,32,34,36,38,40]
    bin_centers_00=0.5*(x00[1:]+x00[:-1])
    axes[0,0].plot(bin_centers_00,n00[0]*100/m, color='black', label="Scénariii fixes sans surventilation nocturne")
    axes[0,0].plot(bin_centers_00,n00[1]*100/m, color='b', label="Scénariii fixes avec surventilation nocturne")
    axes[0,0].plot(bin_centers_00,n00[2]*100/m, color='r',linestyle=':', label="NoMASS (seed 0)")
    axes[0,0].plot(bin_centers_00,n00[3]*100/m, color='g',linestyle=':', label="NoMASS (seed 1000)")
    bin_centers_10=0.5*(x10[1:]+x10[:-1])
    axes[1,0].plot(bin_centers_10,n10[0]*100/m, color='black')
    axes[1,0].plot(bin_centers_10,n10[1]*100/m, color='b')
    axes[1,0].plot(bin_centers_10,n10[2]*100/m, color='r',linestyle=':')
    axes[1,0].plot(bin_centers_10,n10[3]*100/m, color='g',linestyle=':')
    bin_centers_01=0.5*(x01[1:]+x01[:-1])
    axes[0,1].plot(bin_centers_01,n01[0]*100/m, color='black')
    axes[0,1].plot(bin_centers_01,n01[1]*100/m, color='b')
    axes[0,1].plot(bin_centers_01,n01[2]*100/m, color='r',linestyle=':')
    axes[0,1].plot(bin_centers_01,n01[3]*100/m, color='g',linestyle=':')
    bin_centers_11=0.5*(x11[1:]+x11[:-1])
    axes[1,1].plot(bin_centers_11,n11[0]*100/m, color='black')
    axes[1,1].plot(bin_centers_11,n11[1]*100/m, color='b')
    axes[1,1].plot(bin_centers_11,n11[2]*100/m, color='r',linestyle=':')
    axes[1,1].plot(bin_centers_11,n11[3]*100/m, color='g',linestyle=':')
    fig.legend()
    axes[0,0].set_title("CH1")
    axes[1,0].set_title("CH3")
    axes[0,1].set_title("CH2")
    axes[1,1].set_title("RDC")
    axes[0,0].set_ylabel("% annuel")
    axes[1,0].set_ylabel("% annuel")
    plt.xticks(ticks=t)
    plt.show()
def plot_grouped_bars():
    plt.figure(figsize=(7,7), dpi=300)
    groups = [[1.04, 0.96],
          [1.69, 4.02]]
    group_labels = ["G1", "G2"]
    num_items = len(group_labels)
    # This needs to be a numpy range for xdata calculations
    # to work.
    ind = np.arange(num_items)
    # Bar graphs expect a total width of "1.0" per group
    # Thus, you should make the sum of the two margins
    # plus the sum of the width for each entry equal 1.0.
    # One way of doing that is shown below. You can make
    # The margins smaller if they're still too big.
    margin = 0.05
    width = (1.-2.*margin)/num_items
    s = plt.subplot(1,1,1)
    for num, vals in enumerate(groups):
        print ("plotting: ", vals)
        # The position of the xdata must be calculated for each of the two data series
        xdata = ind+margin+(num*width)
        # Removing the "align=center" feature will left align graphs, which is what
        # this method of calculating positions assumes
        gene_rects = plt.bar(xdata, vals, width)
        # You should no longer need to manually set the plot limit since everything 
        # is scaled to one.
        # Also the ticks should be much simpler now that each group of bars extends from
        # 0.0 to 1.0, 1.0 to 2.0, and so forth and, thus, are centered at 0.5, 1.5, etc.
        s.set_xticks(ind+0.5)
        s.set_xticklabels(group_labels)
def max_temp(df):
    max=df.max()
    #max.to_csv(("df_sans_surv_max.csv"))    
    max=df["ZCH1:Zone Mean Air Temperature [C](TimeStep)"].max()
    index_max=df[df["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]==max].index.values
    solution=df.iloc[index_max]
    print (solution)
def taux_ouverture_fen_deter(df):
    """durée d'ouverture en nombre d'heures des fenetres avec surventilation nocturne sur une année"""
    fen=df.iloc[:, ["Zone Ventilation Air Change Rate [ach](TimeStep)" in col for col in df.columns]]
    df_result=pd.DataFrame(columns=["RDC","CH1","CH2","CH3"])
    for col in fen.columns:
        #df_result[col]=fen[fen[col]>1].count()
        if "RDC" in col:
            df_result["RDC"]=fen[fen[col]>1].count()#[fen[col]>1]
        if "CH1" in col:
            df_result["CH1"]=fen[fen[col]>1].count()[col]
        if "CH2" in col:
            df_result["CH2"]=df[df[col]>1].count()
        if "CH3" in col:
            df_result["CH3"]=df[df[col]>1].count()
        if "SDB" in col:
            df_result["SDB"]=df[df[col]>1].count()
    return df_result*5/60

def plot_nomass_temp(df):
    #fig = plt.figure()
    #fig.set_size_inches(15,10)
    ax=df.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)", figsize=(15,8), yticks=[-10,-5,0,5,10,15,20,25,30])#label="ZCH1:Zone Mean Air Temperature [C](TimeStep)")
    df.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=ax,yticks=[-10,-5,0,5,10,15,20,25,30])
    df.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=ax,yticks=[-10,-5,0,5,10,15,20,25,30])
    ax1=ax.twinx()
    df.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax1,yticks=[0,100,200,300,400,500,600,700,800,900]).legend()
    #ax.xticks(rotation = 70)
    ax.legend(loc="upper left")
    plt.savefig('./graphes/deter_hiver.png')
    plt.show()

def plot_RDC_hiver(nomass1,nomass2,deter):
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne 1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    deter.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(-10,30),c='b',title="Scénarii fixes", label="RDC: Temperature Intérieure[C]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="RDC: Fraction Volet Baissé",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","RDC_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',label="RDC: Eclairage",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_0,c='y', label="RDC: Eclairement Lumineux [lux]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)",ax=ax1_0,c='y',style=":",label="Eclairement Lumineux Extérieur [lux]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","RDC_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="RDC: Nombre Occupants",legend=False)
    
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    nomass1.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(-10,30),c='b',title="NoMASS (seed 0)", label="RDC: Temperature Intérieure[C]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONEBLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="RDC: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONELIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',label="RDC: Eclairage",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_1,c='y', label="RDC: Eclairement Lumineux [lux]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)",ax=ax1_1,c='y',style=":",label="Eclairement Lumineux Extérieur [lux]",legend=False)
    nomass1.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONENUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="RDC: Nombre Occupants",legend=False)
    
    #colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    nomass2.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(-10,30),c='b',title="NoMASS (seed 1000)", label="RDC: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONEBLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="RDC: Fraction Volet Baissé", legend=False)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONELIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',label="RDC: Eclairage").legend(loc="upper left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_2,c='y', label="RDC: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)",ax=ax1_2,c='y',style=":",label="Eclairement Lumineux Extérieur [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","RDCTHERMALZONENUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="RDC: Nombre Occupants").legend(fontsize=7)
    plt.show()
def plot_RDC_ete(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/10
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True, figsize=(30,10))
    #colonne 1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    ax2_0=axes[2,0].twinx()
    deter.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(0,40),c='b',title="Scénarii fixes", label="RDC: Temperature Intérieure[C]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="RDC: Fraction Volet Baissé",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","RDC_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="RDC: Eclairage",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","SURVENTILATION NOCTURNE:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,1],c='b',label="RDC: Surventilation Nocturne",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_0,c='r',style=":",label="RDC: Débit renouvellement d'air [Vol/h] ",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_0,c='y', label="RDC: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDC_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="RDC: Nombre Occupants",legend=False)
    
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    ax2_1=axes[2,1].twinx()
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(0,40),c='b',title="NoMASS (seed 0)", label="RDC: Temperature Intérieure[C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONEBLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="RDC: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONELIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="RDC: Eclairage").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONEWINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,1],c='b',label="RDC: Fenetre").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_1,c='r',style=":",label="RDC: Débit renouvellement d'air [Vol/h] ",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_1,c='y', label="RDC: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONENUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="RDC: Nombre Occupants",legend=False)

    # colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    ax2_2=axes[2,2].twinx()
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(0,40),c='b',title="NoMASS (seed 1000)", label="RDC: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONEBLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="RDC: Fraction Volet Baissé",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONELIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="RDC: Eclairage",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONEWINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,1],c='b',label="RDC: Fenetre",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_2,c='r',style=":",label="RDC: Débit renouvellement d'air [Vol/h] ").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDC THERMAL ZONE:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_2,c='y', label="RDC: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","RDCTHERMALZONENUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="RDC: Nombre Occupants").legend(fontsize=7)
    #plt.MaxNLocator(3)
    #pyplot.locator_params(axis='x', nbins=7)
    plt.savefig(".\graphes\RDC_ete.png")
    plt.show()
def plot_CH_hiver(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/2
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(-10,30),c='b',title="Scénarii fixes", label="ZCH1: Temperature Intérieure[C]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(-10,30),c='r',title="Scénarii fixes", label="ZCH2: Temperature Intérieure[C]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Eclairage",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Eclairage",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_0,c='b', style="-.",label="ZCH1: Eclairement Lumineux [lux]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_0,c='r', style="-.",label="ZCH2: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH1_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="ZCH1: Nombre Occupants",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH1_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='r',label="ZCH2: Nombre Occupants",legend=False)
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(-10,30),c='b',title="NoMASS (seed 0)", label="ZCH1: Temperature Intérieure[C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(-10,30),c='r', label="ZCH2: Temperature Intérieure[C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH1BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH2BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH1LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Eclairage",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH2LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Eclairage",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_1,c='b', style="-.", label="ZCH1: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_1,c='r', style="-.", label="ZCH2: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="ZCH1: Nombre Occupants",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='r',label="ZCH2: Nombre Occupants",legend=False)
    #colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(-10,30),c='b',title="NoMASS (seed 1000)", label="ZCH1: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(-10,30),c='r', label="ZCH2: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH1BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé",legend=False)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH2BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Fraction Volet Baissé",legend=False)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH1LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Eclairage").legend(loc="upper left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH2LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Eclairage").legend(loc="upper left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_2,c='b', style="-.", label="ZCH1: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_2,c='r', style="-.", label="ZCH2: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="ZCH1: Nombre Occupants").legend(fontsize=7,loc="upper right")
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='r',label="ZCH2: Nombre Occupants").legend(fontsize=7,loc="upper right")
    
    plt.show()
def plot_CH_ete(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/10
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    ax2_0=axes[2,0].twinx()
    #GRAPHE1
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(0,40),c='b',title="Scénarii fixes", label="ZCH1: Temperature Intérieure[C]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(0,40),c='r',title="Scénarii fixes", label="ZCH2: Temperature Intérieure[C]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    #GRAPHE2
    deter.iloc[59907:59907+2017].plot("Date/Time","SURVENTILATION NOCTURNE:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,1],c='b',label="Surventilation Nocturne",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_0,c='b',style=":",label="ZCH1: Débit renouvellement d'air [Vol/h]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_0,c='r',style=":",label="ZCH2: Débit renouvellement d'air [Vol/h]",legend=False)
    #GRAPHE3
    nomass.iloc[59907:59907+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="Fraction Volet Baissé",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',label="Eclairage",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_0,c='b', style="-.",label="ZCH1: Eclairement Lumineux [lux]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_0,c='r', style="-.",label="ZCH2: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    #GRAPHE4
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH1_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="ZCH1: Nombre Occupants",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH1_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='r',label="ZCH2: Nombre Occupants",legend=False)
    
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    ax2_1=axes[2,1].twinx()
    #GRAPHE1
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(0,40),c='b',title="NoMASS (seed 0)", label="ZCH1: Temperature Intérieure[C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(0,40),c='r', label="ZCH2: Temperature Intérieure[C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    #GRAPHE2
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,1],c='b',label="ZCH1: Fenetre").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_1,c='b',style=":",label="ZCH1: Débit renouvellement d'air [Vol/h] ",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,1],c='r',label="ZCH2: Fenetre").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_1,c='r',style=":",label="ZCH2: Débit renouvellement d'air [Vol/h] ",legend=False)
    #GRAPHE3
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_1,c='b', style="-.", label="ZCH1: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_1,c='r', style="-.", label="ZCH2: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    #GRAPHE4
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="ZCH1: Nombre Occupants",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='r',label="ZCH2: Nombre Occupants",legend=False)
       
    #colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    ax2_2=axes[2,2].twinx()
    #GRAPHE1
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(0,40),c='b',title="NoMASS (seed 1000)", label="ZCH1: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(0,40),c='r', label="ZCH2: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    #GRAPHE2
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,1],c='b',label="ZCH1: Fenetre",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_2,c='b',style=":",label="ZCH1: Débit renouvellement d'air [Vol/h] ").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,1],c='r',label="ZCH2: Fenetre",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_2,c='r',style=":",label="ZCH2: Débit renouvellement d'air [Vol/h] ").legend(loc="upper right",fontsize=7)
    #GRAPHE3
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2BLINDFRACTION:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',label="ZCH2: Fraction Volet Baissé",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_2,c='b', style="-.", label="ZCH1: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_2,c='r', style="-.", label="ZCH2: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]").legend(loc="upper right",fontsize=7)
    #GRAPHE4
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="ZCH1: Nombre Occupants").legend(fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='r',label="ZCH2: Nombre Occupants").legend(fontsize=7)
    
    plt.show()
def plot_CH3_hiver(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/2
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(-10,30),c='b',title="Scénarii fixes", label="ZCH3: Temperature Intérieure[C]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH3: Fraction Volet Baissé",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH3: Eclairage",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_0,c='b', style="-.",label="ZCH3: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]",legend=False)
    deter.iloc[6338:6338+2017].plot("Date/Time","ZCH3_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="ZCH3: Nombre Occupants",legend=False)
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(-10,30),c='b',title="NoMASS (seed 0)", label="ZCH1: Temperature Intérieure[C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[1,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH1: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH2: Eclairage",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_1,c='b', style="-.", label="ZCH1: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]",legend=False)
    nomass.iloc[6338:6338+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="ZCH3: Nombre Occupants",legend=False)
    #colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(-10,30),c='b',title="NoMASS (seed 1000)", label="ZCH3: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[1,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH3: Fraction Volet Baissé",legend=False)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',label="ZCH3: Eclairage").legend(loc="upper left",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax1_2,c='b', style="-.", label="ZCH3: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[6338:6338+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/2",ax=ax1_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 2 [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[6338:6338+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="ZCH3: Nombre Occupants").legend(fontsize=7,loc="upper right")
    
    plt.show()
def plot_CH3_ete(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/10
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne 1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    ax2_0=axes[2,0].twinx()
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(0,40),c='b',title="Scénarii fixes", label="CH3: Temperature Intérieure[C]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","SURVENTILATION NOCTURNE:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,1],c='b',label="CH3: Surventilation Nocturne",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_0,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_0,c='y', label="CH3: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="CH3: Nombre Occupants",legend=False)
    
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    ax2_1=axes[2,1].twinx()
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(0,40),c='b',title="NoMASS (seed 0)", label="CH3: Temperature Intérieure[C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,1],c='b',label="CH3: Fenetre").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_1,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_1,c='y', label="CH3: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="CH3: Nombre Occupants",legend=False)

    # colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    ax2_2=axes[2,2].twinx()
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(0,40),c='b',title="NoMASS (seed 1000)", label="CH3: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,1],c='b',label="CH3: Fenetre",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_2,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_2,c='y', label="CH3: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="CH3: Nombre Occupants").legend(fontsize=7)
    #plt.MaxNLocator(3)
    #pyplot.locator_params(axis='x', nbins=7)
    #plt.savefig(".\graphes\CH3_ete.png")
    plt.show()
def plot_CH3_ete(nomass,nomass2,deter):
    nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10"]=nomass["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]/10
    fig,axes=plt.subplots(nrows=4,ncols=3,sharex=True)
    #colonne 1
    ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    ax2_0=axes[2,0].twinx()
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,0],ylim=(0,40),c='b',title="Scénarii fixes", label="CH3: Temperature Intérieure[C]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,0],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_0, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","VOLET:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","CH1_ECLAIRAGE:Schedule Value [](TimeStep)",ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","SURVENTILATION NOCTURNE:Schedule Value [](TimeStep)",ax=axes[1,0],yticks=[0,1],c='b',label="CH3: Surventilation Nocturne",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_0,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_0,c='y', label="CH3: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_0,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    deter.iloc[59907:59907+2017].plot("Date/Time","ZCH3_OCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,0],c='b',label="CH3: Nombre Occupants",legend=False)
    
    #colonne 2
    ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    ax2_1=axes[2,1].twinx()
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,1],ylim=(0,40),c='b',title="NoMASS (seed 0)", label="CH3: Temperature Intérieure[C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,1],c='g', style=":",label="Temperature Extérieure [C]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_1, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,1],yticks=[0,1],c='b',label="CH3: Fenetre").legend(loc="upper left",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_1,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_1,c='y', label="CH3: Eclairement Lumineux [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_1,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]",legend=False)
    nomass.iloc[59907:59907+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,1],c='b',label="CH3: Nombre Occupants",legend=False)

    # colonne 3
    ax0_2=axes[0,2].twinx()
    ax1_2=axes[1,2].twinx()
    ax2_2=axes[2,2].twinx()
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Mean Air Temperature [C](TimeStep)",ax=axes[0,2],ylim=(0,40),c='b',title="NoMASS (seed 1000)", label="CH3: Temperature Intérieure[C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",ax=axes[0,2],c='g', style=":",label="Temperature Extérieure [C]").legend(loc="lower left",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",color='y',ax=ax0_2, style=":",label="Site Direct Solar Radiation Rate per Area [W/m2]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='b',alpha=0.7,label="CH3: Fraction Volet Baissé",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3LIGHTSTATE:Schedule Value [](TimeStep)",ax=axes[2,2],yticks=[0,0.25,0.5,0.75,1],c='r',alpha=0.7,label="CH3: Eclairage",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3WINDOWSTATE0:Schedule Value [](TimeStep)",ax=axes[1,2],yticks=[0,1],c='b',label="CH3: Fenetre",legend=False)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Zone Ventilation Air Change Rate [ach](TimeStep)",ax=ax1_2,c='r',style=":",label="CH3: Débit renouvellement d'air [Vol/h] ").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3:Daylighting Reference Point 1 Illuminance [lux](TimeStep)",ax=ax2_2,c='y', label="CH3: Eclairement Lumineux [lux]").legend(loc="upper right",fontsize=7)
    nomass.iloc[59907:59907+2017].plot("Date/Time","Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)/10",ax=ax2_2,c='y',style=":",label="Eclairement Lumineux Extérieur divisé par 10 [lux]").legend(loc="upper right",fontsize=7)
    nomass2.iloc[59907:59907+2017].plot("Date/Time","ZCH3NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)",ax=axes[3,2],c='b',label="CH3: Nombre Occupants").legend(fontsize=7)
    #plt.MaxNLocator(3)
    #pyplot.locator_params(axis='x', nbins=7)
    #plt.savefig(".\graphes\CH3_ete.png")
    plt.show()

def moyenne_glissante_norme (valeurs, intervalle):
    indice_debut=(intervalle - 1) // 2
    liste_moyennes=valeurs[1:intervalle]
    liste_moyennes += [(0.2*valeurs[i - indice_debut]+0.3*valeurs[i - indice_debut+1]+0.4*valeurs[i - indice_debut+2]+
                    0.5*valeurs[i - indice_debut+3]+0.6*valeurs[i - indice_debut+4]+0.8*valeurs[i - indice_debut+5]+
                    valeurs[i - indice_debut+6]) / 3.8 for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def overheating_from_csv(csv_filename):
    zones_areas={"RDC THERMAL ZONE":48.75,"ZCH1":14.86,"ZCH2":10.16,"ZCH3":9.64,"ZSDB":14.09}
    building_area=97.5
    print("computing overheating")
    indoor = None
    out = None
    heures_inconfort=[]
    oh = []
    data=pd.read_csv(csv_filename)
    indoor = data.iloc[:, [
        "Mean Air Temperature" in col for col in data.columns]]
    out = data.iloc[:,[
        "Outdoor Air Drybulb Temperature" in col for col in data.columns]]
    Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
    for zone, area in zones_areas.items():
        oh_zone=0
        heures_inconfort_zone=0
        indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
        T_moy_jour=[float(indoor_zone[i:289+i].mean()) for i in range(0,len(indoor_zone),288)]
        for i in range(len(T_moy_jour)):
            if T_moy_jour[i]>(Tconfort[i]+2):
                oh_zone+=T_moy_jour[i]-(Tconfort[i]+2)
                heures_inconfort_zone+=1
        oh.append(oh_zone)
        heures_inconfort.append(heures_inconfort_zone)
    area_tot=building_area
    areas=[]
    for zone,area in zones_areas.items():
        areas.append(area)
    oh_tot=sum([x*y for x,y in zip(areas,oh)])/area_tot  #somme pondérée par les surfaces
    heures_inconfort_tot=sum([x*y for x,y in zip(areas,heures_inconfort)])/area_tot  
    print("overheating = %s °C/h" % (oh_tot))
    print("heures inconfort = %s " % (heures_inconfort_tot))
    return heures_inconfort_tot
def duree_ouverture(df):
    fen=df.iloc[:, ["WINDOWSTATE0:Schedule Value [](TimeStep)" in col for col in df.columns]]
    df_result=dict()
    '''fen['is_large'] = (fen["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"] > 1)
    fen['crossing'] = (fen.is_large != fen.is_large.shift()).cumsum()
    fen['count'] = fen.groupby(['is_large', 'crossing']).cumcount(ascending=False) + 1
    fen.loc[fen.is_large == False, 'count'] = 0'''
    '''for k, v in result:
        print(f'[group {k}]')
        print(v)
        print('\n')'''
    #print(result.count()["ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)"].tolist())
    for col in fen.columns:
        if "RDC" in col:
            result=fen[fen[col]==1].groupby((fen[col]!= 1).cumsum())
            l=result.count()[col].tolist()
            df_result["RDC"]=[i*5 for i in l]
        if "CH1" in col:
            result=fen[fen[col]==1].groupby((fen[col]!= 1).cumsum())
            l=result.count()[col].tolist()
            df_result["CH1"]=[i*5 for i in l]
        if "CH2" in col:
            result=fen[fen[col]==1].groupby((fen[col]!= 1).cumsum())
            l=result.count()[col].tolist()
            df_result["CH2"]=[i*5 for i in l]
        if "CH3" in col:
            result=fen[fen[col]==1].groupby((fen[col]!= 1).cumsum())
            l=result.count()[col].tolist()
            df_result["CH3"]=[i*5 for i in l]
    print(df_result)
    return df_result

'''fig = plt.figure()
fig.set_size_inches(15,10)
df["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].iloc[5471:5471+2017].plot()
df["ZCH2NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"].iloc[5471:5471+2017].plot()
plt.yticks([0,1,2])
#plt.yticks([0,0.25,0.5,0.75,1])
plt.savefig('./graphes/numberofoccupants.png')
plt.show()'''
#
'''with open("./modelNoMASS/IDM_NoMASS_withYear.csv", "w") as file_out:
    with open ("./modelNoMASS/IDM_NoMASS.csv","r") as file_in:
        for line in file_in:
            file_out.write("2021/"+ line)'''

'''def my_to_datetime(date_str):
    if date_str[8:10] != '24':
        return pd.to_datetime(date_str, format=' %m/%d  %H:%M:%S')
    date_str = date_str[0:8] + '00' + date_str[10:]
    return pd.to_datetime(date_str, format=' %m/%d  %H:%M:%S') + dt.timedelta(days=1)

print(my_to_datetime(' 04/10  24:00:00'))
df=pd.read_csv("./Results_to_Plot/IDM_NoMASS.csv",index_col="Date/Time", parse_dates=True, date_parser=my_to_datetime)
print (df.index)
#df['Date/Time'] = df['Date/Time'].apply(my_to_datetime)
#print(df[' 10/10':' 10/14'])
#pd.to_datetime('04/10 23:00:00', format='%m/%d %H:%M:%S')

df["ZCH3BLINDFRACTION:Schedule Value [](TimeStep) "][:"1900/12/31 23:55:00"].plot()
plt.show()
plt.savefig('./graphes/ZCH3BLINDFRACTION.png')'''

'''for zone in zones:
    for variable in variables:
        df_nomass[zone+variable].plot()
        plt.xticks(rotation = 70,size=10)
        plt.savefig('./graphes/'+zone+'.png')#
        plt.show()
        #plt.clf()'''

#plot_RDC_ete(df_nomass1,df_nomass2,df_deter)
#plot_CH3_ete(df_nomass1,df_nomass2,df_deter)

#plot_RDC_hiver(df_nomass1,df_nomass2,df_deter)
#plot_CH3_hiver(df_nomass1,df_nomass2,df_deter)

#print(taux_occupation_nomass(df_nomass1))
#print(taux_occupation_nomass(df_nomass2))

csv_filename="./Results_to_Plot/IDM.csv"
overheating_from_csv(csv_filename)
#print(taux_fen_nomass_jour(df_nomass1).mean())
#plot_taux_fen(df_nomass1,df_nomass2)
#df_nomass1.hist(column=["ZCH1:Zone Mean Air Temperature [C](TimeStep)","ZCH2:Zone Mean Air Temperature [C](TimeStep)"], bins=[19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],grid=False)
#plot_hist_temp(df_sans_surv,df_deter,df_nomass1, df_nomass2)
#print(temp_max(df_nomass2))
#plot_ch1_2502(df_nomass1)
#plot_occupation(df_nomass1)
#print(heure_fen_ouvert_hors_chauff(df_nomass1))
#print(heure_fen_ouvert_hors_chauff(df_nomass2))
#plot_eclairage_volet(df_nomass1,df_nomass2)
#heure_fen_ouvert_jour(df_nomass1).plot()
#plt.show()
#print(heure_fen_ouvert_jour_annee(df_nomass1))
#print(heure_fen_ouvert_nuit_annee(df_nomass1))

#print(taux_ouverture_fen_deter(df_deter))

#liste=[1, 183, 1, 1, 5, 1, 15, 110, 1, 4, 4, 46, 403, 1, 1, 14, 75, 1, 10, 73, 2, 1, 1, 21, 1, 3, 9, 53, 53, 1, 16, 1, 2, 207, 40, 87, 12, 2, 269, 1, 9, 134, 1, 2, 1, 2]
#d1=duree_ouverture(df_nomass1)
#d2=duree_ouverture(df_nomass2)
#plot_hist_duree(d1,d2)