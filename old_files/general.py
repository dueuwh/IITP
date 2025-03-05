import numpy as np
import pandas as pd

a = [1,4,3,5,2]
a_arr = np.array(a)
a_arr = a_arr.reshape(1, -1)
a_df = pd.DataFrame(a_arr)
a_df.columns = ['a', 'b', 'c', 'd', 'e']
print(a_df)