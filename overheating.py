import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
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
    ################################# initialisation des dataframes
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

    ################################# remplissage de dataframe df avec les données qui nous interessent du csv
    df['Text']=df_NoMASS0.iloc[:,["Outdoor Air Drybulb Temperature" in col for col in df_NoMASS0.columns]].values
    Indoor=df_NoMASS0.iloc[:, ["Mean Air Temperature" in col for col in df_NoMASS0.columns]]
    for zone in zones:
        df['T_'+zone]=Indoor.iloc[:, [zone in col for col in Indoor.columns]].values
    ################################# calcul de Tconfort adaptatif selon la norme NF EN 16798-1
    Text_moy_jour=[float(df["Text"][i:288+i].mean()) for i in range(0,len(df["Text"]),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes]
    #plt.plot(Text_moy_jour, label="Text_moy")
    #plt.plot(Tconfort, label="Tconf")
    Tconf_annee=[]
    Text_gliss_annee=[]
    for i in range (365):
        Tconf_annee+=[Tconfort[i]]*288
    Tconf=pd.DataFrame(Tconf_annee).values
    df_Tconf['Tconf']=Tconf_annee
    '''for i in range (365):
        Text_gliss_annee+=[Text_glissantes[i]]*288
    Text_gliss=pd.DataFrame(Text_gliss_annee).values
    df_Tconf['Textgliss']=Text_gliss_annee'''
    ################################## calcul DH ##############################################################
    df_DH[['RDC','ZCH1','ZCH2','ZCH3','ZSDB']]=df[['T_RDC','T_ZCH1','T_ZCH2','T_ZCH3','T_ZSDB']]-Tconf-2
    df_DH[df_DH<0]=0
    DH=df_DH.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/12 #pour spécifier la période de calcul df_H.loc['7/1/2001 00:00:00':'7/7/2001 23:55:00',]
    df_H=df_DH[df_DH>0]
    df_H[df_H>0]=1
    Hours=df_H.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].sum()/12
    #print(df_H.loc['7/1/2001 00:00:00':'7/7/2001 23:55:00',].resample('1D').sum()/12)
    '''ax=df["T_ZCH1"].loc['1/1/2001 00:00:00':'12/31/2001 23:55:00',].plot()
    df_Tconf.loc['1/1/2001 00:00:00':'12/31/2001 23:55:00',].plot(ax=ax)#["Tconf"]
    plt.legend()
    plt.show()'''
    return DH, Hours
def Discomfort_INCAS(df_DH,df_Hours):
    """calculer inconfort de toute la maison INCAS par pondération des surfaces. prend en entrée le résultat de la fonction overheating
    """
    oh=(df_DH['RDC']*48.75+df_DH["ZCH1"]*14.86+df_DH["ZCH2"]*10.16+df_DH["ZCH3"]*9.64+df_DH["ZSDB"]*14.09)/97.5
    hours=(df_Hours['RDC']*48.75+df_Hours["ZCH1"]*14.86+df_Hours["ZCH2"]*10.16+df_Hours["ZCH3"]*9.64+df_Hours["ZSDB"]*14.09)/97.5
    return oh,hours
def heure_fen_ouvert(csv):
    """durée d'ouverture en nombre d'heures des fenetres annuel"""
    df=pd.read_csv(csv)
    FEN = pd.DataFrame()
    FEN=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    FEN['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    FEN.set_index('date',inplace=True)
    #FEN.plot(subplots=True)
    #plt.show()
    df_result=FEN.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/12#'4/15/2001 00:00:00':'10/14/2001 23:55:00'
    return df_result
def heure_fen_ouvert_occ(csv):
    df=pd.read_csv(csv)
    occupation=df.iloc[:, ["NUMBEROFOCCUPANTS" in col for col in df.columns]]
    occupation[occupation>0]=1
    df=df.iloc[:, ["WINDOWSTATE" in col for col in df.columns]]
    FEN = pd.DataFrame()
    FEN['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    FEN.set_index('date',inplace=True)
    for zone in zones[:-1]:
        FEN['fen_'+zone]=df.iloc[:, [zone in col for col in df.columns]].values
        FEN['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
        FEN.loc[FEN['Occ_'+zone] ==0, 'fen_'+zone] = 0
    #FEN.plot(subplots=True)
    #plt.show()
    df_result=FEN.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',['fen_RDC','fen_ZCH1','fen_ZCH2','fen_ZCH3']].sum()/12#'6/21/2001 00:00:00':'9/22/2001 23:55:00'
    return df_result
def heure_surven_nocturne(csv):
    df=pd.read_csv(csv)
    df=df.iloc[:, ["Zone Ventilation Air Change Rate [ach](TimeStep)" in col for col in df.columns]]
    FEN=pd.DataFrame()
    FEN['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    FEN.set_index('date',inplace=True)
    for zone in zones[:-1]:
        FEN['ach_'+zone]=df.iloc[:, [zone in col for col in df.columns]].values
        FEN.loc[FEN['ach_'+zone] >1, 'ach_'+zone] = 1
        FEN.loc[FEN['ach_'+zone] <1, 'ach_'+zone] = 0
    df_result=FEN.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].sum()/12
    return df_result
def heure_surven_nocturne_occ(csv):
    df=pd.read_csv(csv)
    occupation=df.iloc[:, ["_OCCUPANTS" in col for col in df.columns]]
    occupation[occupation>0]=1
    df=df.iloc[:, ["Zone Ventilation Air Change Rate [ach](TimeStep)" in col for col in df.columns]]
    FEN=pd.DataFrame()
    FEN['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    FEN.set_index('date',inplace=True)
    for zone in zones[:-1]:
        FEN['ach_'+zone]=df.iloc[:, [zone in col for col in df.columns]].values
        FEN['Occ_'+zone]=occupation.iloc[:, [zone in col for col in occupation.columns]].values
        FEN.loc[(FEN['ach_'+zone] >1)&(FEN['Occ_'+zone]==1), 'ach_'+zone] = 1
        FEN.loc[(FEN['ach_'+zone] <1)|(FEN['Occ_'+zone]<1), 'ach_'+zone] = 0
    df_result=FEN.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',['ach_RDC','ach_ZCH1','ach_ZCH2','ach_ZCH3']].sum()/12
    return df_result
def taux_volet(csv):
    df=pd.read_csv(csv)
    V = pd.DataFrame()
    V=df.iloc[:, ["BLINDFRACTION" in col for col in df.columns]]
    E=df.iloc[:, ["LIGHTSTATE" in col for col in df.columns]]
    V.rename(columns={"RDCTHERMALZONEBLINDFRACTION:Schedule Value [](TimeStep)": "RDC",
                    "ZCH1BLINDFRACTION:Schedule Value [](TimeStep)": "CH1",
                    "ZCH2BLINDFRACTION:Schedule Value [](TimeStep)": "CH2",
                    "ZCH3BLINDFRACTION:Schedule Value [](TimeStep) ": "CH3",
                    "ZSDBBLINDFRACTION:Schedule Value [](TimeStep)": "SDB"},inplace=True)
    E.rename(columns={"RDCTHERMALZONELIGHTSTATE:Schedule Value [](TimeStep)": "RDC",
                    "ZCH1LIGHTSTATE:Schedule Value [](TimeStep)": "CH1",
                    "ZCH2LIGHTSTATE:Schedule Value [](TimeStep)": "CH2",
                    "ZCH3LIGHTSTATE:Schedule Value [](TimeStep)": "CH3",
                    "ZSDBLIGHTSTATE:Schedule Value [](TimeStep)": "SDB"},inplace=True)
    V['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    V.set_index('date',inplace=True)
    E['date'] = pd.date_range(start='1/1/2001 00:00:00', end='12/31/2001 23:55:00', freq='5min')
    E.set_index('date',inplace=True)
    V=V.between_time('7:00','21:00').resample('D').mean()
    E=E.between_time('7:00','21:00').resample('D').mean()
    fig,axes=plt.subplots(nrows=5,ncols=1,sharex=True,sharey=True,figsize=(6,3.8))
    zones=['RDC','CH1','CH2','CH3','SDB']
    for i in range(len(zones)):
        V[zones[i]].plot(ax=axes[i],label=zones[i]+"_volets")
        E[zones[i]].plot(ax=axes[i],label=zones[i]+"_eclairage")
        axes[i].legend(fontsize=7)
    plt.savefig("volets_eclairage.png")
    plt.show()
    #df_result=V.loc['6/21/2001 00:00:00':'9/22/2001 23:55:00',].mean()#'4/15/2001 00:00:00':'10/14/2001 23:55:00'
    #return df_result.loc['4/15/2001 00:00:00':'10/14/2001 23:55:00',].mean()
#fichier csv des données
NoMASS0="D:/results_nomass_conv/IDM_NoMASS_0eplus.csv"
NoMASS1000="C:/Users/elkhatts/Desktop/optimisation-NoMASS/modelNoMASS/seed1000_stack/IDM_NoMASS.csv"
NoMASS0_sans_volets="C:/Users/elkhatts/Desktop/optimisation-NoMASS/modelNoMASS/NoMASS_sans_volets/IDM_NoMASS_sans_volets.csv"
IDM="C:/Users/elkhatts/Desktop/IDM/IDM.csv"
IDM_sans_surventilation="C:/Users/elkhatts/Desktop/IDM/IDM_sans_surventilation.csv"
'''df_DH,df_Hours=overheating(IDM)
DH,Hours=Discomfort_INCAS(df_DH,df_Hours)
print(df_DH,df_Hours)
print(DH,Hours)'''
df_DH_conv=pd.DataFrame()
df_Hours_conv=pd.DataFrame()
for i in range (100):
    NoMASS="D:/results_NoMASS_GM/IDM_NoMASS_"+str(i)+"eplus.csv"
    df_DH,df_Hours=overheating(NoMASS)
    df_DH_conv=df_DH_conv.append(df_DH,ignore_index=True)
    df_Hours_conv=df_Hours_conv.append(df_Hours,ignore_index=True)
df_DH_conv.to_csv('DH_GM_conv_ann_total.csv',index=False)
df_Hours_conv.to_csv('Hours_GM_conv_ann_total.csv',index=False)
print(df_DH_conv)
print(df_DH_conv.mean())
print(df_Hours_conv.mean())
'''df_FEN_conv=pd.DataFrame()#durées ouverture fenetres
for i in range (100):
    NoMASS="D:/results_nomass_conv/IDM_NoMASS_"+str(i)+"eplus.csv"
    df=heure_fen_ouvert_occ(NoMASS)
    df_FEN_conv=df_FEN_conv.append(df,ignore_index=True)
print(df_FEN_conv.mean())'''
#print(taux_volet(NoMASS0))
