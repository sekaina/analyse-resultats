import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''blind_fractions_moy_jour=[]
for i in range(0,2880,288):
    moyenne=blind_fractions_RDC[i+84:252+i].mean() 
    print (i,moyenne)
    blind_fractions_moy_jour.append(moyenne)'''

data=pd.read_csv("./Results_To_Plot/IDM_NoMASS.csv")
blind_fractions_nomass=data.iloc[:, ["BLINDFRACTION" in col for col in data.columns]]
volet=data.iloc[:, ["VOLET:Schedule" in col for col in data.columns]]
#print(volet.head())
days_in_months=[31,28,31,30,31,30,31,31,30,31,30,31]
months_index={"janvier":8928,"février":16992,"mars":25920,"avril":34560,"mai":43488,"juin":52128,
                "juillet":61056,"aout":69984,"septembre":78624,"octobre":87552,"novembre":96192,"decembre":105120} #288*days_in_months[i]
print(months_index["janvier"])
volet_moy=[float(volet[i+84:252+i].mean()) for i in range(months_index["janvier"],months_index["février"],288)]
#print(volet_moy, len(volet_moy))
blind_fractions_RDC_nomass=blind_fractions_nomass.iloc[:, ["CH2" in col for col in blind_fractions_nomass.columns]]
blind_fractions_CH1_nomass=blind_fractions_nomass.iloc[:, ["CH1" in col for col in blind_fractions_nomass.columns]]
blind_fractions_moy_RDC_nomass=[float(blind_fractions_RDC_nomass[i+84:252+i].mean()) for i in range(months_index["janvier"],months_index["février"],288)]#(0,8928,288)pour janvier
#print(blind_fractions_moy_RDC_nomass)
blind_fractions_moy_CH1_nomass=[float(blind_fractions_CH1_nomass[i+84:252+i].mean()) for i in range(months_index["janvier"],months_index["février"],288)]#len(blind_fractions_CH1_nomass)
#print(blind_fractions_RDC.iloc[288,0])
#print(len(blind_fractions_RDC[84+288+288+288:252+288+288+288]),blind_fractions_RDC[84+288+288+288:252+288+288+288].mean())
'''y=[0 for i in range (1,29)]
jour_deter=[i for i in range (1,29)]
jour_rdc=[i-0.1 for i in range (1,29)]
jour_ch1=[i+0.1 for i in range (1,29)]
plt.vlines(jour_deter,volet_moy,y, color='g', label="deter")
plt.vlines(jour_rdc,blind_fractions_moy_RDC_nomass,y, color='r', label="RDC")
plt.vlines(jour_ch1,blind_fractions_moy_CH1_nomass,y, color="b", label="CH1")
plt.xticks(np.arange(min(jour_deter), max(jour_deter)+1, 1))'''
jour=[i for i in range (1,29)]
plt.plot(jour,volet_moy, label="deter")
plt.plot(jour,blind_fractions_moy_RDC_nomass,  label="CH2")
plt.plot(jour,blind_fractions_moy_CH1_nomass, label="CH1")
plt.xticks(np.arange(min(jour), max(jour)+1, 1))
plt.title("taux de fermeture journalier (entre 7h et 21h) des volets pour le mois de février")
plt.legend()
plt.savefig("fig.png")
plt.show()

'''

                Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
                Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
                Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
                for zone, area in config.zones_areas.items():
                    oh_zone=0
                    heures_inconfort_zone=0
                    indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
                    T_moy_jour=[float(indoor_zone[i:289+i].mean()) for i in range(0,len(indoor_zone),288)]
                    for i in range(len(T_moy_jour)):
                        if T_moy_jour[i]>(Tconfort[i]+2):
                            oh_zone+=T_moy_jour[i]-(Tconfort[i]+2)
                            heures_inconfort_zone+=1
                    oh.append(oh_zone)
                    heures_inconfort.append(heures_inconfort_zone)'''