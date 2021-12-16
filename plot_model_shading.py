
import matplotlib.pyplot as plt
import numpy as np
#B = blind state (0=ferme) (1=ouvert)
def P_baisser_arr(Ein,B):
    a=-7.41
    bin=0.001035
    b=2.17
    m=np.exp(a+bin*Ein+b*B)
    return m/(1+m)
def P_lever_arr(Ein,B):
    a=-1.52
    bin=-0.000654
    b=-3.139
    m=np.exp(a+bin*Ein+b*B)
    return m/(1+m)
def P_baisser_inter(Ein,B):
    a=-8.013
    bin=0.000841
    b=1.27
    m=np.exp(a+bin*Ein+b*B)
    return m/(1+m)
def P_lever_inter(Ein,B):
    a=-3.625
    bin=-0.000276
    b=-2.683
    m=np.exp(a+bin*Ein+b*B)
    return m/(1+m)
def P_baisser_entier(Eext,B):
    a=-0.27
    bext=0.00000091
    b=-2.23
    m=np.exp(a+bext*Eext+b*B)
    return m/(1+m)
def P_lever_entier(Eext,B):
    a=0.435
    bext=-0.0000231
    b=1.95
    m=np.exp(a+bext*Eext+b*B)
    return m/(1+m)
def fraction_ferme(u,B_init):
    a=np.exp(-2.294+1.522*B_init)
    k=1.708
    return a*(-np.log10(u))**(1/k)

Ein = np.arange(1,3000,1) # start,stop,step
Eext = np.arange(1,50000,1)
B = np.arange(0,1,0.0001)
#P_baisser
'''y1 = [P_baisser_arr(i, 1) for i in Ein]
y7 = [P_baisser_arr(i, 0.5) for i in Ein]
y2 = [P_baisser_inter(i, 1) for i in Ein]
y8 = [P_baisser_inter(i, 0.5) for i in Ein]
plt.figure()
plt.plot(Ein, y1, label="Arrivée et B=1", linestyle="-", c="k")
plt.plot(Ein, y7, label="Arrivée et B=0.5", linestyle="--", c="k")
plt.plot(Ein, y2, label="Intermédiaire et B=1", linestyle="-", c="g")
plt.plot(Ein, y8, label="Intermédiaire et B=0.5", linestyle="--", c="g")
#plt.ylim([0,0.04])
plt.xlabel("Eclairement lumineux sur le plan de travail [lux]")
plt.ylabel("Probabilité de baisser")
plt.legend()
plt.savefig("./graphes/P_baisser.png")
plt.show()'''
#P_lever
'''y3 = [P_lever_arr(i, 0) for i in Ein]
y9 = [P_lever_arr(i, 0.5) for i in Ein]
y4 = [P_lever_inter(i, 0) for i in Ein]
y10 = [P_lever_inter(i,0.5) for i in Ein]

plt.figure()
plt.plot(Ein, y3, label="Arrivée et B=0", linestyle="-", c="k")
plt.plot(Ein, y9, label="Arrivée et B=0.5", linestyle="--", c="k")
plt.plot(Ein, y4, label="Intermédiaire et B=0", linestyle="-", c="g")
plt.plot(Ein, y10, label="Intermédiaire et B=0.5", linestyle="--", c="g")
#plt.ylim([0,0.04])
plt.xlabel("Eclairement lumineux sur le plan de travail [lux]")
plt.ylabel("Probabilité de lever")
plt.legend()
plt.savefig("./graphes/P_lever.png")
plt.show()'''
#P baisser entièrement
'''y5 = [P_baisser_entier(0,i) for i in B]
y11 = [P_baisser_entier(50000,i) for i in B]
plt.figure()
plt.plot(B, y5, label="Eclairement lumineux horizontal extérieur global = 0[lux]", linestyle="-", c="k")
plt.plot(B, y11, label="Eclairement lumineux horizontal extérieur global = 50000[lux]", linestyle="--", c="k")
#plt.ylim([0,0.5])
plt.xlabel("fraction initiale d'ouverture du volet")
plt.ylabel("Probabilité de baisser entièrement")
plt.legend()
plt.savefig("./graphes/P_baisser_entier.png")
plt.show()'''
#P lever entièrement

'''y6 = [P_lever_entier(0,i) for i in B]
y12 = [P_lever_entier(50000,i) for i in B]
plt.figure()
plt.plot(B, y6, label="Eclairement lumineux horizontal extérieur global = 0[lux]", linestyle="-", c="k")
plt.plot(B, y12, label="Eclairement lumineux horizontal extérieur global = 50000[lux]", linestyle="--", c="k")
#plt.ylim([0,0.5])
plt.xlabel("fraction initiale d'ouverture du volet")
plt.ylabel("Probabilité de lever entièrement")
plt.legend()
plt.savefig("./graphes/P_lever_entier.png")
plt.show()'''

#fraction de fermeture
U=np.arange(0.01,1,0.0001)
fraction_25 = [fraction_ferme(i,0.25) for i in U]
fraction_50 = [fraction_ferme(i,0.50) for i in U]
fraction_75 = [fraction_ferme(i,0.75) for i in U]
fraction_100 = [fraction_ferme(i,1) for i in U]
plt.figure()
plt.plot(U, fraction_25, label="B=25%", linestyle="-", c="k")
plt.plot(U, fraction_50, label="B=50%", linestyle="-", c="b")
plt.plot(U, fraction_75, label="B=75%", linestyle="-", c="g")
plt.plot(U, fraction_100, label="B=100%", linestyle="-", c="r")

plt.ylim([0,1])
plt.xlabel("nombre aléatoire")
plt.ylabel("augmentation du taux d'occultation")
plt.legend()
plt.savefig("./graphes/fraction_fermeture.png")
plt.show()

#P_lever+baisser pour comparer
'''y1 = [P_baisser_arr(i, 1) for i in Ein]
y7 = [P_baisser_arr(i, 0.5) for i in Ein]
y2 = [P_baisser_inter(i, 1) for i in Ein]
y8 = [P_baisser_inter(i, 0.5) for i in Ein]
y3 = [P_lever_arr(i, 0) for i in Ein]
y9 = [P_lever_arr(i, 0.5) for i in Ein]
y4 = [P_lever_inter(i, 0) for i in Ein]
y10 = [P_lever_inter(i,0.5) for i in Ein]

plt.figure()
plt.plot(Ein, y3, label="Arrivée et B=0", linestyle="-", c="k")
plt.plot(Ein, y9, label="Arrivée et B=0.5", linestyle="--", c="k")
plt.plot(Ein, y4, label="Intermédiaire et B=0", linestyle="-", c="g")
plt.plot(Ein, y10, label="Intermédiaire et B=0.5", linestyle="--", c="g")
plt.plot(Ein, y1, label="Arrivée et B=1", linestyle="-", c="k")
plt.plot(Ein, y7, label="Arrivée et B=0.5", linestyle="--", c="k")
plt.plot(Ein, y2, label="Intermédiaire et B=1", linestyle="-", c="g")
plt.plot(Ein, y8, label="Intermédiaire et B=0.5", linestyle="--", c="g")
plt.xlabel("Eclairement lumineux sur le plan de travail [lux]")
plt.ylabel("Probabilité de baisser/lever")
plt.legend()
plt.savefig("./graphes/P_arr_inter.png")
plt.show()'''