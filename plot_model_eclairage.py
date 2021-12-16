
import matplotlib.pyplot as plt
import numpy as np
def P_allumage_arr(E):
    a=-0.0175
    b=4.0835
    c=1.0361
    m=1.8223
    '''if (np.log10(E)<=0.843):
        return 1
    elif (np.log10(E)>=2.818):
        return 0
    else :'''
    return a+c/(1+np.exp(b*(np.log10(E)-m)))
def P_allumage_inter(E):
    a=0.0027
    b=64.19
    c=0.017
    m=2.41
    return a+c/(1+np.exp(b*(np.log10(E)-m)))
print(P_allumage_inter(300))
x = np.arange(1,700,1) # start,stop,step
y1 = [P_allumage_arr(i) for i in x]
y2 = [P_allumage_inter(i) for i in x]
plt.figure()
plt.plot(x, y1)
#plt.ylim([0,0.04])
plt.xlabel("Eclairement lumineux [lux]")
plt.ylabel("Probabilité d'allumer l'éclairage artificiel")

plt.savefig("./graphes/P_allumer_arr.png")
plt.show()
