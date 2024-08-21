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

senior = False
deap = True


if deap:
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)
    subject_no = subject_list[0]
    deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
    print("deap_dataset.keys(): ", deap_dataset.keys())
    print("deap data:\n", deap_dataset["data"].shape)
    print("deap labels:\n", deap_dataset["labels"].shape)
    print("deap labels total:\n", deap_dataset["labels"])
    
    for i in range(40):
        plt.plot(deap_dataset["data"][i, 37, :])
        plt.title(f"i:{i}, j:{1} | {deap_dataset['labels'][i, :]}")
        plt.xlim(3000,3500)
        plt.show()
        
        
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




        