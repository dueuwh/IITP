import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt




def split_ecg():
    base_path = "D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/filtered_ecg_signals.csv"
    file = pd.read_csv(base_path, index_col=0)
    for index in file.index.unique().tolist():
        select_index = np.array(deepcopy(file.loc[index]))
        select_index = select_index[:, 1]
        np.save(f"D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/ecg_{index}.npy", select_index)

if __name__ == "__main__":
    split_ecg()