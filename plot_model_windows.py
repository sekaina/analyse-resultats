
import matplotlib.pyplot as plt
import numpy as np

aop = 2.151
bopout = 0.172
shapeop = 0.418
#P01arr
a01arr = -13.88
b01inarr = 0.312
b01outarr = 0.0433
b01absprevarr = 1.862
b01rnarr = -0.45
#P01int
a01int = -12.23
b01inint = 0.281
b01outint = 0.0271
b01presint = -0.000878
b01rnint = -0.336
#P01dep
a01dep = -8.75
b01outdep = 0.1371
b01absdep = 0.84
b01gddep = 0.83
#P10dep
a10dep = -8.54
b10indep = 0.213
b10outdep = -0.0911
b10absdep = 1.614
b10gddep = -0.923
def P_ouvrir_arr(indoorTemperature, outdoorTemperature, previousDuration, rain):  
    m=np.exp(a01arr + b01inarr * indoorTemperature +b01outarr * outdoorTemperature +b01rnarr * rain +b01absprevarr *previousDuration)
    return m/(1+m)

def P_ouvrir_inter(indoorTemperature, outdoorTemperature, currentDuration, rain):  
    m=np.exp(a01int + b01inint * indoorTemperature + b01outint * outdoorTemperature +b01presint * currentDuration +b01rnint * rain)
    return m/(1+m)
def P_fermer_depart(indoorTemperature, dailyMeanTemperature, durationLongerThanEightHours, groundFloor):  
    m=np.exp(a10dep + b10indep * indoorTemperature +b10outdep * dailyMeanTemperature +b10absdep * durationLongerThanEightHours +b10gddep * groundFloor)
    return m/(1+m)
def P_ouvrir_depart(dailyMeanTemperature, durationLongerThanEightHours, groundFloor):  
    m=np.exp(a01dep + b01outdep * dailyMeanTemperature +b01absdep * durationLongerThanEightHours +b01gddep * groundFloor)
    return m/(1+m)
def duree(u,Text):
    a=np.exp(2.151+0.172*Text)
    k=0.418
    return a*(-np.log10(u))**(1/k)

Tin = np.arange(10,40,1) # start,stop,step
Text_moy = np.arange(0,40,1)
#P_ouvrir_arrivee
'''y1 = [P_ouvrir_arr(i, 0, 1,0) for i in Tin]
y2 = [P_ouvrir_arr(i, 0, 0,0) for i in Tin]
y7 = [P_ouvrir_arr(i, 0, 0,1) for i in Tin]
y8 = [P_ouvrir_arr(i, 20, 0,1) for i in Tin]
y3 = [P_ouvrir_arr(i, 20, 1,0) for i in Tin]
y4 = [P_ouvrir_arr(i, 20, 0,0) for i in Tin]
y5 = [P_ouvrir_arr(i, 40, 1,0) for i in Tin]
y6 = [P_ouvrir_arr(i, 40, 0,0) for i in Tin]

plt.figure()
plt.plot(Tin, y1, label="Text=0°C, ap_abs=1 et pluie=0", linestyle="-", c="k")
plt.plot(Tin, y2, label="Text=0°C, ap_abs=0 et pluie=0", linestyle="--", c="k")
plt.plot(Tin, y7, label="Text=0°C, ap_abs=0 et pluie=1", linestyle=":", c="k")
plt.plot(Tin, y3, label="Text=20°C, ap_abs=1 et pluie=0", linestyle="-", c="r")
plt.plot(Tin, y4, label="Text=20°C, ap_abs=0 et pluie=0", linestyle="--", c="r")
plt.plot(Tin, y8, label="Text=20°C, ap_abs=0 et pluie=1", linestyle=":", c="r")
plt.plot(Tin, y5, label="Text=40°C, ap_abs=1 et pluie=0", linestyle="-", c="b")
plt.plot(Tin, y6, label="Text=40°C, ap_abs=0 et pluie=0", linestyle="--", c="b")
#plt.ylim([0,0.04])
plt.xlabel("Température intérieure [°C]")
plt.ylabel("Probabilité d'ouvrir la fenêtre en arrivée")
plt.legend()
plt.savefig("./graphes/P_ouvrir_arrive.png")
plt.show()'''
#P_ouvrir_inter
'''y1 = [P_ouvrir_inter(i, 0, 1800,0) for i in Tin]
y2 = [P_ouvrir_inter(i, 20, 1800,0) for i in Tin]
y3 = [P_ouvrir_inter(i, 40, 1800,0) for i in Tin]
y4 = [P_ouvrir_inter(i, 0, 3600,0) for i in Tin]
y5 = [P_ouvrir_inter(i, 20, 3600,0) for i in Tin]
y6 = [P_ouvrir_inter(i, 40, 3600,0) for i in Tin]
plt.figure()
plt.plot(Tin, y1, label="Text=0°C et Dpres=0,5h", linestyle="-", c="k")
plt.plot(Tin, y2, label="Text=20°C et Dpres=0,5h", linestyle="--", c="k")
plt.plot(Tin, y3, label="Text=40°C et Dpres=0,5h", linestyle="-.", c="k")
plt.plot(Tin, y4, label="Text=0°C et Dpres=1h", linestyle="-", c="r")
plt.plot(Tin, y5, label="Text=20°C et Dpres=1h", linestyle="--", c="r")
plt.plot(Tin, y6, label="Text=40°C et Dpres=1h", linestyle="-.", c="r")
#plt.ylim([0,0.04])
plt.xlabel("Température intérieure [°C]")
plt.ylabel("Probabilité d'ouvrir la fenêtre en période intermédiaire")
plt.legend()
plt.savefig("./graphes/P_ouvrir_inter.png")
#plt.show()
#P_ouvrir_depart
y1 = [P_ouvrir_depart(i, 1,0) for i in Text_moy]
y2 = [P_ouvrir_depart(i, 1,1) for i in Text_moy]
y3 = [P_ouvrir_depart(i, 0,0) for i in Text_moy]
y4 = [P_ouvrir_depart(i, 0,1) for i in Text_moy]
plt.figure()
plt.plot(Text_moy, y1, label="av_abs=1 et RDC=0", linestyle="-", c="k")
plt.plot(Text_moy, y2, label="av_abs=1 et RDC=1", linestyle="--", c="k")
plt.plot(Text_moy, y3, label="av_abs=0 et RDC=0", linestyle="-", c="r")
plt.plot(Text_moy, y4, label="av_abs=0 et RDC=1", linestyle="--", c="r")
#plt.ylim([0,0.04])
plt.xlabel("Température extérieure glissante sur une journée [°C]")
plt.ylabel("Probabilité d'ouvrir la fenêtre en départ")
plt.legend()
plt.savefig("./graphes/P_ouvrir_depart.png")
plt.show()'''
#P_fermer_depart
'''y1 = [P_fermer_depart(10,i, 0,1) for i in Text_moy]
y6 = [P_fermer_depart(10,i, 0,0) for i in Text_moy]
y8 = [P_fermer_depart(10,i, 1,1) for i in Text_moy]
y2 = [P_fermer_depart(20,i, 0,1) for i in Text_moy]
y3 = [P_fermer_depart(30,i, 0,1) for i in Text_moy]
y7 = [P_fermer_depart(30,i, 0,0) for i in Text_moy]
y9 = [P_fermer_depart(30,i, 1,1) for i in Text_moy]
y4 = [P_fermer_depart(20,i, 0,0) for i in Text_moy]
y5 = [P_fermer_depart(20,i, 1,1) for i in Text_moy]
plt.figure()
plt.plot(Text_moy, y1, label="Tint=10°C, av_abs=0 et RDC=1", linestyle="-", c="k")
plt.plot(Text_moy, y6, label="Tint=10°C, av_abs=0 et RDC=0", linestyle="-", c="r")
plt.plot(Text_moy, y8, label="Tint=10°C, av_abs=1 et RDC=1", linestyle="-", c="b")
plt.plot(Text_moy, y2, label="Tint=20°C, av_abs=0 et RDC=1", linestyle="--", c="k")
plt.plot(Text_moy, y4, label="Tint=20°C, av_abs=0 et RDC=0", linestyle="--", c="r")
plt.plot(Text_moy, y5, label="Tint=20°C, av_abs=1 et RDC=1", linestyle="--", c="b")
plt.plot(Text_moy, y3, label="Tint=30°C, av_abs=0 et RDC=1", linestyle="-.", c="k")
plt.plot(Text_moy, y7, label="Tint=30°C, av_abs=0 et RDC=0", linestyle="-.", c="r")
plt.plot(Text_moy, y9, label="Tint=30°C, av_abs=1 et RDC=1", linestyle="-.", c="b")
#plt.ylim([0,0.04])
plt.xlabel("Température extérieure glissante sur une journée [°C]")
plt.ylabel("Probabilité de fermer la fenêtre en départ")
plt.legend()
plt.savefig("./graphes/P_fermer_depart.png")
plt.show()'''
#duree d'ouverture
U=np.arange(0.01,1,0.0001)
y5 = [duree(i,5) for i in U]
y30 = [duree(i,30) for i in U]
plt.figure()
plt.plot(U, y5, label="Text=5°C", linestyle="-", c="k")
plt.plot(U, y30, label="Text=30°C", linestyle="--", c="k")
plt.xlabel("nombre aléatoire")
plt.ylabel("durée d'ouverture en minutes")
plt.ylim([0,1000])
plt.legend()
plt.savefig("./graphes/duree_ouverture.png")
plt.show()