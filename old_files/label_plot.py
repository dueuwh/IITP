# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 03:36:27 2024

@author: ys
"""

import os
import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
import pickle
sys.path.append("D:/home/BCML/drax/emma_code/")
from SP_test_LIVE_mp import BPM
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

senior = False
deap = True
illuminance_2 = False

if illuminance_2:
    base_dir = "D:/home/BCML/drax/data/illuminance_2/label/"
    folder_list = os.listdir(base_dir)
    sel_folder = folder_list[0]
    
    sel_dir = base_dir+sel_folder+'/'
    file_list = [name for name in os.listdir(sel_dir+'/') if '.txt' in name]
    
    label_ppg = []
    with open(sel_dir+file_list[0], 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            label_ppg.append(float(line.split()[0]))
            
    plt.plot(label_ppg, label="ppg", color='r')
    # plt.xlim(1000,1200)
    plt.legend()
    plt.show()

if deap:
    # DEAP ppg label index: 38
    # DEAP ppg sampling rate: 128 Hz
    # DEAP label (valence, arousal, dominance, liking)
    fs = 128
    
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)
    total_dataset_x = []
    total_dataset_y_a = []
    total_dataset_y_v = []
    total_dataset_y_d = []
    for subject_no in subject_list:
        deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
        # print("deap_dataset.keys(): ", deap_dataset.keys())
        # print("deap data:\n", deap_dataset["data"].shape)
        # print("deap labels:\n", deap_dataset["labels"].shape)
        # print("deap labels total:\n", deap_dataset["labels"])
        row = deap_dataset["data"].shape[-1]
        save_df = pd.DataFrame(np.nan, index=range(row), columns=range(40))
        
        for i in range(40):
            total_dataset_x.append(deap_dataset["data"][i, 38, :])
            total_dataset_y_a.append(deap_dataset["labels"][i, 1])
            total_dataset_y_v.append(deap_dataset["labels"][i, 0])
            total_dataset_y_d.append(deap_dataset["labels"][i, 2])
            
    total_dataset_x = np.array(total_dataset_x)
    total_dataset_y_a = np.array(total_dataset_y_a)
    total_dataset_y_v = np.array(total_dataset_y_v)
    total_dataset_y_d = np.array(total_dataset_y_d)
    
    test_start = 900
    
    train_dataset_x = total_dataset_x[:test_start, :]
    train_dataset_y_a = total_dataset_y_a[:test_start]
    train_dataset_y_d = total_dataset_y_d[:test_start]
    train_dataset_y_v = total_dataset_y_v[:test_start]
    
    test_dataset_x = total_dataset_x[test_start: , :]
    test_dataset_y_a = total_dataset_y_a[test_start:]
    test_dataset_y_d = total_dataset_y_d[test_start:]
    test_dataset_y_v = total_dataset_y_v[test_start:]
    
    model_arousal = RandomForestRegressor(n_estimators=200, random_state=0)
    model_arousal.fit(train_dataset_x, train_dataset_y_a)
    
    model_valence = RandomForestRegressor(n_estimators=200, random_state=0)
    model_valence.fit(train_dataset_x, train_dataset_y_v)
    
    model_dominance = RandomForestRegressor(n_estimators=200, random_state=0)
    model_dominance.fit(train_dataset_x, train_dataset_y_d)
    
    y_pred_a = model_arousal.predict(test_dataset_x)
    y_pred_v = model_valence.predict(test_dataset_x)
    y_pred_d = model_dominance.predict(test_dataset_x)
    
    y_a_mae = mean_absolute_error(test_dataset_y_a, y_pred_a)
    y_v_mae = mean_absolute_error(test_dataset_y_v, y_pred_v)
    y_d_mae = mean_absolute_error(test_dataset_y_d, y_pred_d)
    
    y_a_rmse = np.sqrt(mean_squared_error(test_dataset_y_a, y_pred_a))
    y_d_rmse = np.sqrt(mean_squared_error(test_dataset_y_d, y_pred_d))
    y_v_rmse = np.sqrt(mean_squared_error(test_dataset_y_v, y_pred_v))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(y_pred_a, y_pred_d, y_pred_v, label="prediction")
    ax.scatter(test_dataset_y_a, test_dataset_y_d, test_dataset_y_v, label="label")
    ax.set_title("prediction 3d scatter")
    plt.legend()
    plt.show()
    
    
        # plt.scatter(deap_dataset["labels"][:, 0], deap_dataset["labels"][:, 1])
        # plt.xlim(0, 10)
        # plt.ylim(0, 10)
        # plt.show()
        
        # for i in range(40):
        #     fig, ax = plt.subplots(2,1)
        #     ax[0].plot(deap_dataset["data"][i, 38, :])
        #     ax[0].set_title(f"subject_number: {subject_no} i:{0}, j:{i} | {deap_dataset['labels'][i, :]} label.shape: {deap_dataset['labels'].shape}")
        #     ax[1].scatter(deap_dataset["labels"][i, 0], deap_dataset["labels"][i, 1])
        #     ax[1].set_title("label")
        #     # plt.xlim(3000,3200)
        #     plt.show()
        
        
        
        
        
        # To do
        # 1. DEAP ppg to bpm
        # 2. Extracting bpm from DEAP dataset using my_pipeline
        # 3. rPPG accuracy
        # 4. HRV parameters
        # 5. LLM classfication
        # 6. End to end (video input, emotion output) model benchmark
    
if senior:
    print("cwd: ", os.getcwd())
    base_dir = "./data/labels/"
    raw_list = os.listdir(base_dir)
    label_list = [name for name in raw_list if '.txt' in name]
    label_dic = {}
    
    for label in label_list:
        label_dic[label] = {0:[], 1:[]}
        
    for label in label_list:
        line_0 = []
        line_1 = []
        with open(base_dir+label, 'r') as f:
            for line in f:
                temp = line.strip().split()
                line_0.append(float(temp[0]))
                line_1.append(float(temp[1]))
        plt.plot(line_0, label="col 0")
        plt.plot(line_1, label="col 1")
        plt.legend()
        plt.title(f"{label} plot")
        plt.show()
        label_dic[label][0] = line_0
        label_dic[label][1] = line_1
    
    # col 1 = button, col 2 = ppg




        