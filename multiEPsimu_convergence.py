import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
def heating_needs_from_csv(table_csv):
    with open(table_csv, 'r') as f:
    # Créer un objet csv à partir du fichier
        obj = np.array(list(csv.reader(f)))
        Chauffage=obj[49][13]
        eclairage=obj[51][2]
        metabolique=obj[1841][8]
    return float(eclairage)
def moyenne_glissante_norme (valeurs, intervalle):
    indice_debut=(intervalle - 1) // 2
    liste_moyennes=valeurs[1:intervalle]
    liste_moyennes += [(0.2*valeurs[i - indice_debut]+0.3*valeurs[i - indice_debut+1]+0.4*valeurs[i - indice_debut+2]+
                    0.5*valeurs[i - indice_debut+3]+0.6*valeurs[i - indice_debut+4]+0.8*valeurs[i - indice_debut+5]+
                    valeurs[i - indice_debut+6]) / 3.8 for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def overheating_from_csv(csv_filename):
    oh = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])#,index=range(1,366)
    oh['date'] = pd.date_range(start='1/1/2000', periods=365, freq='D')
    oh.set_index('date',inplace=True)
    heures_inconfort = pd.DataFrame(columns=['date','RDC','ZCH1','ZCH2','ZCH3','ZSDB'])#,index=range(1,366)
    heures_inconfort['date'] = pd.date_range(start='1/1/2000', periods=365, freq='D')
    heures_inconfort.set_index('date',inplace=True)
    print("computing overheating")
    indoor = None
    data=pd.read_csv(csv_filename)
    indoor = data.iloc[:, [
        "Mean Air Temperature" in col for col in data.columns]]
    out = data.iloc[:,[
        "Outdoor Air Drybulb Temperature" in col for col in data.columns]]
    Text_moy_jour=[float(out[i:288+i].mean()) for i in range(0,len(out),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
    for zone in oh.columns:
        plt.figure()
        oh_zone=[]
        heures_inconfort_zone=[]
        for col in indoor.columns:
            if zone in col: 
                indoor_zone=indoor[col]
        #indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
        indoor_zone=indoor_zone.values.tolist()
        plt.plot(indoor_zone, label="indoor_zone")
        plt.plot(Tconfort, label="Tconf")
        plt.legend()
        plt.show()
        #T_moy_jour=[float(indoor_zone[i:288+i].mean()) for i in range(0,len(indoor_zone),288)]
        for i in range(365):
            oh_jour=0
            heures_inconfort_jour=0
            for j in range(288*i,288+288*i):
                if indoor_zone[j]>(Tconfort[i]+2):#(Tconfort[i]+2):
                    oh_jour+=(indoor_zone[j]-(Tconfort[i]+2))/12#(indoor_zone[j]-(Tconfort[i]+2))
                    heures_inconfort_jour+=1/12
            oh_zone.append(oh_jour) # 24*TIMESTEP
            heures_inconfort_zone.append(heures_inconfort_jour)
        oh[zone]=oh_zone
        heures_inconfort[zone]=heures_inconfort_zone
    #ajouter colonne inconfort de la maison
    oh["oh_tot"]=(oh['RDC']*48.75+oh["ZCH1"]*14.86+oh["ZCH2"]*10.16+oh["ZCH3"]*9.64+oh["ZSDB"]*14.09)/97.5
    print(oh.sum())
    heures_inconfort["heures_inconfort_tot"]=(heures_inconfort['RDC']*48.75+heures_inconfort["ZCH1"]*14.86+heures_inconfort["ZCH2"]*10.16+heures_inconfort["ZCH3"]*9.64+heures_inconfort["ZSDB"]*14.09)/97.5
    print(heures_inconfort.sum())
    print(oh)
    #ax=oh['ZCH1'].plot(label="ZCH1")
    #oh['ZCH2'].plot(label="ZCH2")
    #ax.set_ylabel("Tzone-(Tconf+2)")
    #plt.plot(oh['ZCH2'])
    #heures_inconfort['ZCH1'].plot(label="ZCH1")
    #zones_areas={"RDC":48.75,"ZCH1":14.86,"ZCH2":10.16,"ZCH3":9.64,"ZSDB":14.09}
    #building_area=97.5
    '''area_tot=building_area
    areas=[]
    for zone,area in zones_areas.items():
        areas.append(area)
    oh_tot=sum([x*y for x,y in zip(areas,oh)])/area_tot  #somme pondérée par les surfaces
    heures_inconfort_tot=sum([x*y for x,y in zip(areas,heures_inconfort)])/area_tot  
    print("overheating = %s °C/h" % (oh_tot))
    print("heures inconfort = %s " % (heures_inconfort_tot))
    return heures_inconfort_tot,oh_tot
    oh.at['2000-01-01','ZCH1']=0
    oh.at['2000-01-01','ZCH2']=0
    oh['RDC'][oh['RDC'] < 0] = 0'''

    
#df = pd.DataFrame(columns=['hours_discomfort','DH','indicateur3'],index=range(0, 100))
#sur l'année
#for i in range(100):
    #table_csv="D:/results/IDM_NoMASS_"+str(i)+"eplus.csv"#
    #df['hours_discomfort'][i],df['DH'][i]=overheating_from_csv(table_csv)
#pour chaque jour (seed0)
table_csv="D:/results/IDM_NoMASS_0eplus-table.csv"#
IDM_SF="./Results_To_Plot/IDM.csv"
IDM_sans_surventilation="C:/Users/elkhatts/Desktop/IDM/IDM_sans_surventilation.csv"
#overheating_from_csv(IDM_sans_surventilation)
#overheating_eachzone_eachday(table_csv)
#print(df)
eclairage=[]
for i in range (100):
    table_csv="D:/results_NoMASS_sans_volets/IDM_NoMASS_"+str(i)+"eplus-table.csv"
    eclairage.append(heating_needs_from_csv(table_csv))
print(eclairage)
print(np.mean(eclairage))