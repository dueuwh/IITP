import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import natsort
import neurokit2 as nk
from sklearn.metrics import mean_absolute_error as mae

"""
Result figure plan

    >>> Haar cascade
    Intra subject
    |subject name     | |subject name     | |subject name     |
    |precision, recall| |precision, recall| |precision, recall|
    |confusion matrix | |confusion matrix | |confusion matrix |
    ___________________________________________________________
    hr accuracy (MAE) high -> low
    
    
    Inter subject
    |subject name     | |subject name     | |subject name     |
    |precision, recall| |precision, recall| |precision, recall|
    |confusion matrix | |confusion matrix | |confusion matrix |
    ___________________________________________________________
    hr accuracy (MAE) high -> low
    
    
    >>> Mediapipe
    Intra subject
    |subject name     | |subject name     | |subject name     |
    |precision, recall| |precision, recall| |precision, recall|
    |confusion matrix | |confusion matrix | |confusion matrix |
    ___________________________________________________________
    hr accuracy (MAE) high -> low
    
    
    Inter subject
    |subject name     | |subject name     | |subject name     |
    |precision, recall| |precision, recall| |precision, recall|
    |confusion matrix | |confusion matrix | |confusion matrix |
    ___________________________________________________________
    hr accuracy (MAE) high -> low    
"""


if __name__ == "__main__":
    bvp_path = "C:/Users/U/Desktop/BCML/IITP/IITP_emotions/data/senior/sychro/results/rppg_toolbox/bvp/unsupervised/"
    label_path = "C:/Users/U/Desktop/BCML/IITP/IITP_emotions/data/senior/sychro/results/rppg_toolbox/total/datafiles/"
    
    #_____________________________________________________________
    # 1      2        3              4      5     6        7
    # anger, anxiety, embarrassment, happy, hurt, neutral, sadness
    #_____________________________________________________________
    
    emotions_idx = [1, 4, 6, 7]
    emotions_names = {1:'anger', 4:'happy', 6:'neutral', 7:'sadness'}
    
    bvp_algorithms = os.listdir(bvp_path)
    common_list = os.listdir(f"{bvp_path}{bvp_algorithms[0]}")
    common_list = natsort.natsorted(common_list)
    bvp_common_list = [name for name in common_list if 'bvp' in name]
    prehr_common_list = [name for name in common_list if 'pre_hr' in name]
    gthr_common_list = [name for name in common_list if 'gt_hr' in name]
    label_list = [name for name in os.listdir(label_path) if "label" in name]
    label_list = natsort.natsorted(label_list)
    
    def gather_dataset():
        label_dic = {}
        rppg_dic = {}
        prehr_dic = {}
        gthr_dic = {}

        for idx, label in enumerate(label_list):
            subject = int(label.split('_')[0][:-1])
            emotion = int(label.split('_')[0][-1])
            if emotion in emotions_idx:
                if subject not in label_dic.keys():
                    label_dic[subject] = {}
                    rppg_dic[subject] = {}
                    prehr_dic[subject] = {}
                    gthr_dic[subject] = {}

                    label_dic[subject][emotions_names[emotion]] = [label]
                    rppg_dic[subject][emotions_names[emotion]] = [bvp_common_list[idx]]
                    prehr_dic[subject][emotions_names[emotion]] = [prehr_common_list[idx]]
                    gthr_dic[subject][emotions_names[emotion]] = [gthr_common_list[idx]]
                else:
                    label_dic[subject][emotions_names[emotion]].append(label)
                    rppg_dic[subject][emotions_names[emotion]].append(bvp_common_list[idx])
                    prehr_dic[subject][emotions_names[emotion]].append(prehr_common_list[idx])
                    gthr_dic[subject][emotions_names[emotion]].append(gthr_common_list[idx])
                
        return label_dic, rppg_dic, prehr_dic, gthr_dic
    
    label_dic, rppg_dic, prehr_dic, gthr_dic = gather_dataset()
    
    def hr_plot():
        for subject in label_dic.keys():
            label_data = []
            for i, file_name in enumerate(label_dic[subject]):
                if i == 0:
                    label_data = np.load(f"{label_path}{label_dic[subject][i]}")
                else:
                    label_data = np.concatenate((label_data, np.load(f"{label_path}{label_dic[subject][i]}")))
            plt.plot(label_data)
            plt.title(f"{subject}")
            plt.show()
    
    hr_plot()
                
                
                
                