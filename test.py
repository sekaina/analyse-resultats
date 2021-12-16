import numpy as np
import pandas as pd
a = np.array([[1, 7, 3], [4, 5, 6]])
df = pd.DataFrame(data=a, index=["row1", "row2"], columns=["column1", "column2","column3"])
print(df.min())