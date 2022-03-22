import pandas as pd
import matplotlib.pyplot as plt 
def moyenne_glissante_norme (valeurs, intervalle):
    indice_debut=(intervalle - 1) // 2
    liste_moyennes=valeurs[1:intervalle]
    liste_moyennes += [(0.2*valeurs[i - indice_debut]+0.3*valeurs[i - indice_debut+1]+0.4*valeurs[i - indice_debut+2]+
                    0.5*valeurs[i - indice_debut+3]+0.6*valeurs[i - indice_debut+4]+0.8*valeurs[i - indice_debut+5]+
                    valeurs[i - indice_debut+6]) / 3.8 for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
zones=['RDC','ZCH1','ZCH2','ZCH3','ZSDB']
def overheating(csv):
    """" returns two dataframes. the first contains DH_year for each zone. the second contains Hours_year for each zone. 
    example DH_year
    RDC      6608.764945
    ZCH1     6838.415686
    ZCH2     6069.539004
    ZCH3     3997.620813
    ZSDB    10155.234836"""
    df_NoMASS0=pd.read_csv(csv)
    #initialisation des dataframes
    df = pd.DataFrame(columns=['date','Text','T_RDC','T_ZCH1','T_ZCH2','T_ZCH3','T_ZSDB'])
    df['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df.set_index('date',inplace=True)

    df_DH = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])
    df_DH['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_DH.set_index('date',inplace=True)

    df_H = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])
    df_H['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_H.set_index('date',inplace=True)

    df_Tconf = pd.DataFrame(columns=['date','Tconf'])
    df_Tconf['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_Tconf.set_index('date',inplace=True)

#remplissage de dataframe df avec les données qui nous interessent du csv
    df['Text']=df_NoMASS0.iloc[:,["Outdoor Air Drybulb Temperature" in col for col in df_NoMASS0.columns]].values
    Indoor=df_NoMASS0.iloc[:, ["Mean Air Temperature" in col for col in df_NoMASS0.columns]]
    occupation=df_NoMASS0.iloc[:, ["NUMBEROFOCCUPANTS" in col for col in df_NoMASS0.columns]]
    occupation[occupation>0]=1
    for zone in zones:
        df['T_'+zone]=Indoor.iloc[:, [zone in col for col in Indoor.columns]].values
        df['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
#calcul de Tconfort adaptatif selon la norme NF EN 16798-1
    Text_moy_jour=[float(df["Text"][i:288+i].mean()) for i in range(0,len(df["Text"]),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes]
    Tconf_annee=[]
    for i in range (365):
        Tconf_annee+=[Tconfort[i]]*288
    Tconf=pd.DataFrame(Tconf_annee).values
    df_Tconf['Tconf']=Tconf_annee
    #df_Tconf.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].plot()
    #plt.show()
#calcul DH
    df_DH[['RDC']]=df[['T_RDC']]-Tconf-2
    for zone in zones:
        df_DH[[zone]]=(df[['T_'+zone]]-Tconf-2)
        df_DH.loc[df['Occ_'+zone] ==0, zone] = 0
    df_DH[df_DH<0]=0
    DH=df_DH.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/12 #pour spécifier la période de calcul df_H.loc['7/1/2001 00:00:00':'7/7/2001 23:55:00',]
    df_H=df_DH[df_DH>0]
    df_H[df_H>0]=1
    Hours=df_H.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/12
    return DH, Hours
def overheating_SF(csv,csv2):
    """" returns two dataframes. the first contains DH_year for each zone. the second contains Hours_year for each zone. 
    example DH_year
    RDC      6608.764945
    ZCH1     6838.415686
    ZCH2     6069.539004
    ZCH3     3997.620813
    ZSDB    10155.234836"""
    df_SF=pd.read_csv(csv)
    df_SF2=pd.read_csv(csv2)#pour obtenir les SF d'occupation pour le cas sans surventilation
    #initialisation des dataframes
    df = pd.DataFrame(columns=['date','Text','T_RDC','T_ZCH1','T_ZCH2','T_ZCH3','T_ZSDB'])
    df['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df.set_index('date',inplace=True)

    df_DH = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])
    df_DH['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_DH.set_index('date',inplace=True)

    df_H = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])
    df_H['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_H.set_index('date',inplace=True)

    df_Tconf = pd.DataFrame(columns=['date','Tconf'])
    df_Tconf['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    df_Tconf.set_index('date',inplace=True)

#remplissage de dataframe df avec les données qui nous interessent du csv
    df['Text']=df_SF.iloc[:,["Outdoor Air Drybulb Temperature" in col for col in df_SF.columns]].values
    Indoor=df_SF.iloc[:, ["Mean Air Temperature" in col for col in df_SF.columns]]
    occupation=df_SF2.iloc[:, ["_OCCUPANTS" in col for col in df_SF2.columns]]
    occupation[occupation>0]=1
    for zone in zones:
        df['T_'+zone]=Indoor.iloc[:, [zone in col for col in Indoor.columns]].values
        df['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
#calcul de Tconfort adaptatif selon la norme NF EN 16798-1
    Text_moy_jour=[float(df["Text"][i:288+i].mean()) for i in range(0,len(df["Text"]),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes]
    Tconf_annee=[]
    for i in range (365):
        Tconf_annee+=[Tconfort[i]]*288
    Tconf=pd.DataFrame(Tconf_annee).values
    df_Tconf['Tconf']=Tconf_annee
#calcul DH
    df_DH[['RDC']]=df[['T_RDC']]-Tconf-2
    for zone in zones:
        df_DH[[zone]]=(df[['T_'+zone]]-Tconf-2)
        df_DH.loc[df['Occ_'+zone] ==0, zone] = 0
    df_DH[df_DH<0]=0
    DH=df_DH.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/12 #pour spécifier la période de calcul df_H.loc['7/1/2001 00:00:00':'7/7/2001 23:55:00',]
    df_H=df_DH[df_DH>0]
    df_H[df_H>0]=1
    Hours=df_H.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/12
    return DH, Hours
def occupation_hours_SF(csv):
    df_SF=pd.read_csv(csv)
    Occ = pd.DataFrame()
    Occ['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    Occ.set_index('date',inplace=True)
    occupation=df_SF.iloc[:, ["_OCCUPANTS" in col for col in df_SF.columns]]
    occupation[occupation>0]=1
    for zone in zones:
        Occ['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
    occupation_hours=Occ.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/12
    #occupation_hours=occupation.loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].sum()/12
    print(occupation_hours)
    return occupation_hours
def occupation_hours_NoMASS(csv):
    df_SF=pd.read_csv(csv)
    Occ = pd.DataFrame()
    Occ['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    Occ.set_index('date',inplace=True)
    occupation=df_SF.iloc[:, ["NUMBEROFOCCUPANTS" in col for col in df_SF.columns]]
    for zone in zones:
        Occ['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
        Occ.loc[Occ['Occ_'+zone]>0, 'Occ_'+zone] = 1
    occupation_hours=Occ.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/12
    #occupation_hours=occupation.loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].sum()/12
    #print(occupation_hours)
    return occupation_hours
def Discomfort_INCAS(df_DH,df_Hours):
    """calculer inconfort de toute la maison INCAS par pondération des surfaces. prend en entrée le résultat de la fonction overheating
    """
    oh=(df_DH['RDC']*48.75+df_DH["ZCH1"]*14.86+df_DH["ZCH2"]*10.16+df_DH["ZCH3"]*9.64+df_DH["ZSDB"]*14.09)/97.5
    hours=(df_Hours['RDC']*48.75+df_Hours["ZCH1"]*14.86+df_Hours["ZCH2"]*10.16+df_Hours["ZCH3"]*9.64+df_Hours["ZSDB"]*14.09)/97.5
    return oh,hours
def plot_ZCH1(SF_sans,SF_avec,NoMASS0,NoMASS1000):
    #lecture des fichiers
    df_SF_sans=pd.read_csv(SF_sans)
    df_SF_avec=pd.read_csv(SF_avec)
    df_NoMASS0=pd.read_csv(NoMASS0)
    df_NoMASS1000=pd.read_csv(NoMASS1000)
    #remplissage dataframe
    SF_sans = pd.DataFrame()
    SF_sans['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    SF_sans.set_index('date',inplace=True)
    SF_avec = pd.DataFrame()
    SF_avec['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    SF_avec.set_index('date',inplace=True)
    NoMASS0 = pd.DataFrame()
    NoMASS0['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    NoMASS0.set_index('date',inplace=True)
    NoMASS1000 = pd.DataFrame()
    NoMASS1000['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    NoMASS1000.set_index('date',inplace=True)
    #conditions extérieures
    NoMASS0["Eclairement Lumineux Extérieur divisé par 10 [lux]"]=df_NoMASS0.loc[:,["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]].values/10
    NoMASS0["Text"]=df_NoMASS0.loc[:,["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]].values
    NoMASS0["Solar_Radiation"]=df_NoMASS0.loc[:,["Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"]].values
    #SF_avec
    SF_avec["Occ_ZCH1"]=df_SF_avec.loc[:,["ZCH1_OCCUPANTS:Schedule Value [](TimeStep)"]].values
    SF_avec["T_ZCH1"]=df_SF_avec.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    SF_avec["Air_ZCH1"]=df_SF_avec.loc[:,["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"]].values
    SF_avec["Scenario_surven"]=df_SF_avec.loc[:,["SURVENTILATION NOCTURNE:Schedule Value [](TimeStep)"]].values
    SF_avec["Scenario_volet"]=df_SF_avec.loc[:,["VOLET:Schedule Value [](TimeStep)"]].values
    SF_avec["Scenario_eclairage"]=df_SF_avec.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    SF_avec["illumi_ZCH1"]=df_SF_avec.loc[:,["ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)"]].values
    #SF_sans
    SF_sans["T_ZCH1"]=df_SF_sans.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    #NoMASS0
    NoMASS0["Occ_ZCH1"]=df_NoMASS0.loc[:,["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"]].values
    NoMASS0["T_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    NoMASS0["Air_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"]].values
    NoMASS0["Scenario_FENETRE"]=df_NoMASS0.loc[:,["ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)"]].values
    NoMASS0["Scenario_volet"]=df_NoMASS0.loc[:,["ZCH1BLINDFRACTION:Schedule Value [](TimeStep)"]].values
    NoMASS0["Scenario_eclairage"]=df_NoMASS0.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    NoMASS0["illumi_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)"]].values
    #NoMASS1000
    NoMASS1000["Occ_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"]].values
    NoMASS1000["T_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    NoMASS1000["Air_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"]].values
    NoMASS1000["Scenario_FENETRE"]=df_NoMASS1000.loc[:,["ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)"]].values
    NoMASS1000["Scenario_volet"]=df_NoMASS1000.loc[:,["ZCH1BLINDFRACTION:Schedule Value [](TimeStep)"]].values
    NoMASS1000["Scenario_eclairage"]=df_NoMASS1000.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    NoMASS1000["illumi_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)"]].values
    #Tconfort
    Text_moy_jour=[float(NoMASS0["Text"][i:288+i].mean()) for i in range(0,len(NoMASS0["Text"]),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes]
    Tconf_annee=[]
    for i in range (365):
        Tconf_annee+=[Tconfort[i]]*288
    NoMASS0['Tconf']=Tconf_annee
    '''for zone in zones:
        SF_avec.rename(columns = {zone+'_OCCUPANTS:Schedule Value [](TimeStep)': 'Occ_'+zone},inplace=True)
        #Occ['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
        SF_avec.loc[SF_avec['Occ_'+zone]>0, 'Occ_'+zone] = 1'''
    #plot
    fig,axes=plt.subplots(nrows=4,ncols=2,sharex=True,figsize=(16,9))
    #colonne1
    #ax0_0=axes[0,0].twinx()
    ax1_0=axes[1,0].twinx()
    #ax2_0=axes[2,0].twinx()
    #GRAPHE1
    #NoMASS0["Solar_Radiation"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax0_0,color='y', style=":",label="Rayonnement solaire [W/m2]")
    NoMASS0["Text"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,0],c='g', style=":",label="Text [C]", title="Scénarii fixes (avec et sans surventilation)")
    NoMASS0['Tconf'].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,0],c='gray', label="T_confort [C]")
    SF_avec["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,0],c='b', label="T_avec_surv [C]")
    SF_sans["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,0],c='r', label="T_sans_surv [C]")
    axes[0,0].set_ylim(0, 40)
    axes[0,0].legend(loc="lower left")
    #ax0_0.legend(loc="upper right")
    #GRAPHE2
    SF_avec["Air_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax1_0,c='b',style=":",label="Q_air [Vol/h]")
    SF_avec["Scenario_surven"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[1,0],yticks=[0,1],c='b',label="FEN")
    axes[1,0].legend(loc="upper left")
    ax1_0.legend(loc="upper right")
    #GRAPHE3
    SF_avec["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='b',label="Fraction Volet Baissé")
    #SF_avec["Scenario_eclairage"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2,0],yticks=[0,0.25,0.5,0.75,1],c='r',label="Eclairage")
    #SF_avec["illumi_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax2_0,c='b', style="-.",label="E_Lumineux [lux]")
    #NoMASS0["Eclairement Lumineux Extérieur divisé par 10 [lux]"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax2_0,c='y',style=":",label="E_Lumineux_Ext/10 [lux]")
    #ax2_0.legend(loc="upper right",fontsize=7)
    axes[2,0].legend(loc="upper left")
    #GRAPHE4
    SF_avec["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3,0],c='b',label="N_Occupants").legend(loc="upper left",fontsize=7)
    axes[3,0].legend(loc="upper left")
    #colonne 2
    #ax0_1=axes[0,1].twinx()
    ax1_1=axes[1,1].twinx()
    #ax2_1=axes[2,1].twinx()
    #GRAPHE1
    #NoMASS0["Solar_Radiation"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax0_1,color='y', style=":",label="Rayonnement solaire [W/m2]")
    NoMASS0["Text"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,1],c='g', style=":",label="Text [C]", title="NoMASS (seed0 et seed1000)")
    NoMASS0['Tconf'].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,1],c='gray', label="T_confort [C]")
    NoMASS0["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,1],c='b', label="T_seed0 [C]")
    NoMASS1000["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0,1],c='r', label="T_seed1000 [C]")
    axes[0,1].set_ylim(0, 40)
    axes[0,1].legend(loc="lower left")
    #ax0_1.legend(loc="upper right")
    #GRAPHE2
    NoMASS0["Air_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax1_1,c='b',style=":",label="Q_air_seed0 [Vol/h]")
    NoMASS0["Scenario_FENETRE"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[1,1],yticks=[0,1],c='b',label="FEN_seed0")
    NoMASS1000["Air_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax1_1,c='r',style=":",label="Q_air_seed1000 [Vol/h]")
    NoMASS1000["Scenario_FENETRE"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[1,1],yticks=[0,1],c='r',label="FEN_seed1000")
    ax1_1.legend(loc="upper right")
    axes[1,1].legend(loc="upper left")
    
    #GRAPHE3
    NoMASS0["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='b',label="Fraction Volet Baissé seed0")
    #NoMASS0["illumi_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax2_1,c='b', style="-.",label="E_Lumineux_seed0 [lux]",legend=False)
    NoMASS1000["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2,1],yticks=[0,0.25,0.5,0.75,1],c='r',label="Fraction Volet Baissé seed1000")
    #NoMASS1000["illumi_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax2_1,c='r', style="-.",label="E_Lumineux_seed1000 [lux]",legend=False)
    #NoMASS0["Eclairement Lumineux Extérieur divisé par 10 [lux]"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax2_1,c='y',style=":",label="E_Lumineux_Ext/10 [lux]",legend=False)
    axes[2,1].legend(loc="upper left")
    #ax2_1.legend(loc="upper right",fontsize=7)
    #GRAPHE4
    NoMASS0["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3,1],c='b',label="N_Occupants_seed0")
    NoMASS1000["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3,1],c='r',label="N_Occupants_seed1000")
    axes[3,1].legend(loc="upper left")
    plt.legend()
    plt.savefig("dynamique_CH1.png")
    plt.show()
def plot_ZCH1_without_surven(SF,SF_avec,NoMASS0,NoMASS1000):
    #lecture des fichiers
    df_SF=pd.read_csv(SF)
    df_SF_avec=pd.read_csv(SF_avec)#juste pour les scénarios de présence
    df_NoMASS0=pd.read_csv(NoMASS0)
    df_NoMASS1000=pd.read_csv(NoMASS1000)
    #remplissage dataframe
    SF = pd.DataFrame()
    SF['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    SF.set_index('date',inplace=True)
    SF_avec = pd.DataFrame()
    SF_avec['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    SF_avec.set_index('date',inplace=True)
    NoMASS0 = pd.DataFrame()
    NoMASS0['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    NoMASS0.set_index('date',inplace=True)
    NoMASS1000 = pd.DataFrame()
    NoMASS1000['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    NoMASS1000.set_index('date',inplace=True)
    #conditions extérieures
    NoMASS0["Eclairement Lumineux Extérieur divisé par 10 [lux]"]=df_NoMASS0.loc[:,["Environment:Site Exterior Horizontal Sky Illuminance [lux](TimeStep)"]].values/10
    NoMASS0["Text"]=df_NoMASS0.loc[:,["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]].values
    NoMASS0["Solar_Radiation"]=df_NoMASS0.loc[:,["Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"]].values
    #SF_avec
    SF_avec["Occ_ZCH1"]=df_SF_avec.loc[:,["ZCH1_OCCUPANTS:Schedule Value [](TimeStep)"]].values
    SF_avec["Scenario_volet"]=df_SF_avec.loc[:,["VOLET:Schedule Value [](TimeStep)"]].values
    SF_avec["Scenario_eclairage"]=df_SF_avec.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    #SF
    SF["T_ZCH1"]=df_SF.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    #NoMASS0
    NoMASS0["Occ_ZCH1"]=df_NoMASS0.loc[:,["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"]].values
    NoMASS0["T_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    NoMASS0["Air_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"]].values
    NoMASS0["Scenario_FENETRE"]=df_NoMASS0.loc[:,["ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)"]].values
    NoMASS0["Scenario_volet"]=df_NoMASS0.loc[:,["ZCH1BLINDFRACTION:Schedule Value [](TimeStep)"]].values
    NoMASS0["Scenario_eclairage"]=df_NoMASS0.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    NoMASS0["illumi_ZCH1"]=df_NoMASS0.loc[:,["ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)"]].values
    #NoMASS1000
    NoMASS1000["Occ_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1NUMBEROFOCCUPANTS:Schedule Value [](TimeStep)"]].values
    NoMASS1000["T_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Zone Mean Air Temperature [C](TimeStep)"]].values
    NoMASS1000["Air_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Zone Ventilation Air Change Rate [ach](TimeStep)"]].values
    NoMASS1000["Scenario_FENETRE"]=df_NoMASS1000.loc[:,["ZCH1WINDOWSTATE0:Schedule Value [](TimeStep)"]].values
    NoMASS1000["Scenario_volet"]=df_NoMASS1000.loc[:,["ZCH1BLINDFRACTION:Schedule Value [](TimeStep)"]].values
    NoMASS1000["Scenario_eclairage"]=df_NoMASS1000.loc[:,["CH1_ECLAIRAGE:Schedule Value [](TimeStep)"]].values
    NoMASS1000["illumi_ZCH1"]=df_NoMASS1000.loc[:,["ZCH1:Daylighting Reference Point 1 Illuminance [lux](TimeStep)"]].values
    #Tconfort
    Text_moy_jour=[float(NoMASS0["Text"][i:288+i].mean()) for i in range(0,len(NoMASS0["Text"]),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes]
    Tconf_annee=[]
    for i in range (365):
        Tconf_annee+=[Tconfort[i]]*288
    NoMASS0['Tconf']=Tconf_annee
    #plot
    fig,axes=plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(8,5),gridspec_kw={'height_ratios': [2,1,1,1]})
    
    #GRAPHE1
    NoMASS0["Text"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0],c='g', style=":",label="Text [C]")
    NoMASS0['Tconf'].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0],c='gray', label="T_confort [C]")
    NoMASS0["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0],c='b', label="TCH1_seed0 [C]")
    NoMASS1000["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0],c='r', label="TCH1_seed1000 [C]")
    SF["T_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[0],c='green', label="TCH1_SF [C]")
    axes[0].set_ylim(0, 45)
    axes[0].legend(loc="lower left", fontsize=7)
    #GRAPHE2
    NoMASS0["Scenario_FENETRE"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[1],yticks=[0,1],c='b',label="Etat_fenêtre_seed0")
    NoMASS1000["Scenario_FENETRE"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[1],yticks=[0,1],c='r',label="Etat_fenêtre_seed1000")
    axes[1].legend(loc="upper left", fontsize=7)
    ax1=axes[1].twinx()
    NoMASS0["Air_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax1,c='b',style=":",label="Débit renouvellement d'air_seed0 [Vol/h]")
    NoMASS1000["Air_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=ax1,c='r',style=":",label="Débit renouvellement d'air_seed1000 [Vol/h]")
    ax1.legend(loc="upper right", fontsize=7)
    #GRAPHE3
    NoMASS0["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2],yticks=[0,0.25,0.5,0.75,1],c='b',label="Fraction Baissée du Volet seed0")
    NoMASS1000["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2],yticks=[0,0.25,0.5,0.75,1],c='r',label="Fraction Baissée du Volet seed1000")
    SF_avec["Scenario_volet"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[2],yticks=[0,0.25,0.5,0.75,1],c='green',label="Fraction Baissée du Volet SF")
    axes[2].legend(loc="upper left", fontsize=7)
    #GRAPHE4
    NoMASS0["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3],c='b',label="Nombre d'occupants_seed0")
    NoMASS1000["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3],c='r',label="Nombre d'occupants_seed1000")
    SF_avec["Occ_ZCH1"].loc['7/28/2001 00:00:00':'8/3/2001 23:55:00',].plot(ax=axes[3],c='green',label="Nombre d'occupants_SF").legend(loc="upper left",fontsize=7)
    axes[3].legend(loc="upper left", fontsize=7)
    #plt.legend()
    plt.savefig("dynamique_CH1_sans_survent.png")
    plt.show()
def gains(csv):
    df=pd.read_csv(csv)
    fenetre=df.iloc[:, ["Zone Windows Total Transmitted Solar Radiation Energy" in col for col in df.columns]]
    sensible_people=df.iloc[:, ["People Sensible Heating Energy" in col for col in df.columns]]
    eclairage=df.iloc[:, ["Lights Total Heating Energy" in col for col in df.columns]]
    taux_occupation=df.iloc[:, ["People Occupant Count" in col for col in df.columns]]
    sensible_people['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    sensible_people.set_index('date',inplace=True)
    eclairage['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    eclairage.set_index('date',inplace=True)
    fenetre['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    fenetre.set_index('date',inplace=True)
    print("gains métaboliques kwh")
    print(sensible_people.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/3600000)#'4/15/2001 00:00:00':'10/14/2001 23:55:00''6/21/2001 00:00:00':'9/22/2001 23:55:00'
    print("gains éclairage kwh")
    print(eclairage.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/3600000)#.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',]
    print("gains fenetres kwh")
    #print(fenetre.sum()/3600000)
    #print(taux_occupation.sum())
def gains_nomass(csv,annee=False,hors_chauffe=False,ete=False):
    df=pd.read_csv(csv)
    fenetre=df.iloc[:, ["Zone Windows Total Transmitted Solar Radiation Energy" in col for col in df.columns]]
    sensible_people=df.iloc[:, ["People Sensible Heating Energy" in col for col in df.columns]]
    eclairage=df.iloc[:, ["Lights Total Heating Energy" in col for col in df.columns]]
    taux_occupation=df.iloc[:, ["People Occupant Count" in col for col in df.columns]]
    sensible_people['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    sensible_people.set_index('date',inplace=True)
    eclairage['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    eclairage.set_index('date',inplace=True)
    fenetre['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    fenetre.set_index('date',inplace=True)
    if annee==True:
        print("annee")
        GM=sensible_people.sum()/3600000
        eclairage=eclairage.sum()/3600000
        fenetre=fenetre.sum()/3600000
    if hors_chauffe==True:
        print("hors_chauffe")
        GM=sensible_people.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/3600000
        eclairage=eclairage.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/3600000
        fenetre=fenetre.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/3600000
    if ete==True:
        print("ete")
        GM=sensible_people.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/3600000
        eclairage=eclairage.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/3600000
        fenetre=fenetre.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/3600000
    return GM,eclairage,fenetre
#fichier csv des données
NoMASS="C:/Users/elkhatts/Desktop/optimisation-NoMASS/modelNoMASS/IDM_NoMASS.csv"
NoMASS0="D:/results_nomass_conv/IDM_NoMASS_0eplus.csv"
NoMASS1000="C:/Users/elkhatts/Desktop/optimisation-NoMASS/modelNoMASS/seed1000_stack/IDM_NoMASS.csv"
NoMASS0_sans_volets="C:/Users/elkhatts/Desktop/optimisation-NoMASS/modelNoMASS/NoMASS_sans_volets/IDM_NoMASS_sans_volets.csv"
IDM="C:/Users/elkhatts/Desktop/IDM/IDM.csv"
IDM_sans_surventilation="C:/Users/elkhatts/Desktop/IDM/IDM_sans_surventilation.csv"

'''df_DH,df_Hours=overheating_SF(IDM_sans_surventilation,IDM)
print(df_DH,df_Hours)
DH,Hours=Discomfort_INCAS(df_DH,df_Hours)
print(DH,Hours)'''
'''df_DH_conv=pd.DataFrame()
df_Hours_conv=pd.DataFrame()
for i in range (100):
    NoMASS="D:/results_NoMASS_GM/IDM_NoMASS_"+str(i)+"eplus.csv"
    df_DH,df_Hours=overheating(NoMASS)
    df_DH_conv=df_DH_conv.append(df_DH,ignore_index=True)
    df_Hours_conv=df_Hours_conv.append(df_Hours,ignore_index=True)
df_DH_conv.to_csv('DH_GM_conv_ann_occ.csv',index=False)
df_Hours_conv.to_csv('Hours_GM_conv_ann_occ.csv',index=False)
print(df_DH_conv.mean())
print(df_Hours_conv.mean())'''
'''df_Occ_conv=pd.DataFrame()
for i in range (100):
    NoMASS="D:/results_nomass_conv/IDM_NoMASS_"+str(i)+"eplus.csv"
    df_Occ=occupation_hours_NoMASS(NoMASS)
    df_Occ_conv=df_Occ_conv.append(df_Occ,ignore_index=True)
print(df_Occ_conv.mean())'''
#occupation_hours_SF(IDM)
#occupation_hours_SF(IDM)
#occupation_hours_NoMASS(NoMASS1000)
#df_DH,df_Hours=overheating_SF(IDM_sans_surventilation,IDM)
#print(df_DH,df_Hours)
#plot_ZCH1(IDM_sans_surventilation,IDM,NoMASS0,NoMASS1000)
#plot_ZCH1_without_surven(IDM_sans_surventilation,IDM,NoMASS0,NoMASS1000)
#gains(IDM_sans_surventilation)
df_GM_conv=pd.DataFrame()
df_Eclairage_conv=pd.DataFrame()
df_Windows_conv=pd.DataFrame()
for i in range (100):
    NoMASS="D:/results_nomass_conv/IDM_NoMASS_"+str(i)+"eplus.csv"
    df_GM,df_Eclairage,df_Windows=gains_nomass(NoMASS,annee=True,hors_chauffe=False,ete=False)
    df_GM_conv=df_GM_conv.append(df_GM,ignore_index=True)
    df_Eclairage_conv=df_Eclairage_conv.append(df_Eclairage,ignore_index=True)
    df_Windows_conv=df_Windows_conv.append(df_Windows,ignore_index=True)
df_GM_conv.to_csv('GM_conv_ann_occ.csv',index=False)
df_Eclairage_conv.to_csv('Eclairage_conv_ann_occ.csv',index=False)
df_Windows_conv.to_csv('Windows_conv_ann_occ.csv',index=False)
print(df_GM_conv.mean())
print(df_Eclairage_conv.mean())
print(df_Windows_conv.mean())