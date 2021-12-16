import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#activities=np.array([[1,1,1,2,5,1]])
activities=np.array([[1,2,3,4,5,6],[1,1,1,2,5,1]])
#activities=pd.read_csv("sex1famstat2edtry3age3computer1civstat1unemp0retired0_activities.csv", header=None)
#print(activities.head())
print(activities[1])
fig,ax=plt.subplots()
plt.imshow(activities)
#ax.xaxis(activities[0])
plt.colorbar()
plt.show()