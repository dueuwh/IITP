import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import pickle
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import shap
import xgboost as xgb
import seaborn as sns
import itertools
import sys

def deap_rppg_loader(verbose=True):
    base_dir = "D:/home/BCML/IITP/data/DEAP/rppg/lgi/"
    label_dir = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(base_dir)
    
    total_rppg = []
    total_label = []
    
    for subject in subject_list:
        folder_list = os.listdir(base_dir+subject)
        for j, folder in enumerate(folder_list):
            sample = f"D:/home/BCML/IITP/data/DEAP/rppg/lgi/{subject}/{folder}/ppg_omit.txt"
            
            data = []
            
            with open(sample, 'r') as f:
                lines = f.readlines()
                temp_step = []
                for line in lines:
                    
                    if '[' in line:
                        temp_step = []
                        temp = line.split('[')[1]
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
                    elif ']' in line:
                        temp = line.split(']')[0]
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
                        data.append(temp_step)
                    else:
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
            crop = []
            for i, window in enumerate(data):
                crop.append(window[-1])
            
            if verbose:
                plt.rcParams.update({'font.size': 20})
                plt.figure(figsize=(200,20))
                plt.plot(crop, marker='o', markersize=6, markerfacecolor='red', markeredgecolor='red')
                plt.title("crop")
                plt.show()
            
            label_load = pickle.load(open(label_dir + subject + '.dat', 'rb'), encoding='latin1')["labels"][j, :]
            
            total_rppg.append(crop)
            total_label.append(label_load)
        
    total_rppg = pd.DataFrame(np.array(total_rppg).T)
    total_label = pd.DataFrame(total_label)
    output = {}
    output["data"] = total_rppg
    output["labels"] = total_label
    return output

def ppg_loader4deap(base_dir, file_name):
    ppg = pd.read_csv(base_dir + "ppg/" + file_name, index_col=0)
    label = pd.read_csv(base_dir + "label/" + file_name, index_col=0)
    
    # The sampling rate of DEAP dataset is 128
    
    return ppg, label, 128.0


def rppg_loader4deap(file_name):
    pass

def standardization():
    pass

def robustscalar():
    pass

def minmaxscalar(inputseries):
    value_max = max(inputseries)
    value_min = min(inputseries)
    value_v = value_max - value_min
    
    return [(value - value_min)/value_v for value in inputseries]


def cube(ax, point1, point2, color):
    
    ax.plot_surface(np.array([[point1[0], point2[0]], [point1[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point1[2], point1[2]], [point1[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point1[0], point2[0]], [point1[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point2[2]], [point2[2], point2[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point1[0], point1[0]], [point1[0], point1[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point2[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point1[0], point1[0]]]), 
                    np.array([[point1[1], point1[1]], [point1[1], point1[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point1[0], point1[0]]]), 
                    np.array([[point2[1], point2[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
 

def D3Scatter(class_num, border_points,
              total_arousal, total_dominance, total_valence, title):
    
    plt.rcParams['font.size'] = 20

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.zeros((len(total_arousal), 3))
    colors[:, 0] = minmaxscalar(total_valence)
    colors[:, 1] = minmaxscalar(total_arousal)
    colors[:, 2] = minmaxscalar(total_dominance)
    
    norm_x = Normalize(vmin=min(total_valence), vmax=max(total_valence))
    norm_y = Normalize(vmin=min(total_arousal), vmax=max(total_arousal))
    norm_z = Normalize(vmin=min(total_dominance), vmax=max(total_dominance))
    
    cmap_x = plt.cm.Reds
    cmap_y = plt.cm.Blues
    cmap_z = plt.cm.Greens
    
    axins_x = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.60, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_y = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.65, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_z = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.70, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    
    cbar_x = fig.colorbar(plt.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_x)
    cbar_y = fig.colorbar(plt.cm.ScalarMappable(norm=norm_y, cmap=cmap_y), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_y)
    cbar_z = fig.colorbar(plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_z)
    
    cbar_x.set_label('Valence (X)')
    cbar_y.set_label('Arousal (Y)')
    cbar_z.set_label('Dominance (Z)')
    
    scatter = ax.scatter(total_valence, total_arousal, total_dominance, s=50, c=colors, alpha=0.6)
    
    if class_num == 2:
        cube(ax, border_points["positive"][0], border_points["positive"][1], 'r')
        cube(ax, border_points["negative"][0], border_points["negative"][1], 'b')
    
    elif class_num == 3:
        cube(ax, border_points["positive"][0], border_points["positive"][1], 'r')
        cube(ax, border_points["neutral"][0], border_points["neutral"][1], 'g')
        cube(ax, border_points["negative"][0], border_points["negative"][1], 'b')
    
    elif class_num == 4:
        cube(ax, border_points["happy"][0], border_points["happy"][1], 'r')
        cube(ax, border_points["anger"][0], border_points["anger"][1], 'm')
        cube(ax, border_points["sadness"][0], border_points["sadness"][1], 'b')
        cube(ax, border_points["calm"][0], border_points["calm"][1], 'g')
    
    ax.view_init(elev=25, azim=-30, vertical_axis='y')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_zlim(-1, 11)
    
    ax.set_title(title)
    ax.set_xlabel("Valence (X)")
    ax.set_ylabel("Arousal (Y)")
    ax.set_zlabel("Dominance (Z)")
    
    plt.show()


def clear_emotion_binary(criterion_small, criterion_big, remote):
    save_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion_binary/"
    rppg_folder = save_dir + "rppg/"
    ppg_folder = save_dir + "ppg/"
    label_folder = save_dir + "label/"
    
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)

    total_arousal = []
    total_valence = []
    total_dominacne = []
    
    clear_label = []
    clear_ppg = []
    clear_rppg = []
    
    p_num = n_num = 0
    
    for subject_no in subject_list:
        if remote:
            deap_dataset = deap_rppg_loader(verbose=False)
        else:
            deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
        
        for i in range(40):
            total_arousal.append(deap_dataset["labels"][i, 1])
            total_valence.append(deap_dataset["labels"][i, 0])
            total_dominacne.append(deap_dataset["labels"][i, 2])
            
            if deap_dataset["labels"][i, 0] > criterion_big:
                p_num += 1
                clear_label.append(0)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
            
            if deap_dataset["labels"][i, 0] < criterion_small:
                n_num += 1
                clear_label.append(1)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
    
    clear_label_df = pd.DataFrame(clear_label)
    clear_label_df.to_csv(label_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    
    if remote:
        clear_rppg_df = pd.DataFrame(clear_rppg)
        clear_rppg_df.to_csv(rppg_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    else:
        clear_ppg_df = pd.DataFrame(clear_ppg)
        clear_ppg_df.to_csv(ppg_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    
    scatter_title = (f"DEAP dataset label distribution\n"
                     f"Negative < {criterion_small}, Positive > {criterion_big}\n"
                     f"RED: Positive-{p_num}, BLUE: Negative-{n_num}")
    
    border_points = {}
    border_points["positive"] = [[criterion_big, 0, 0], [9, 9, 9]]
    border_points["negative"] = [[0, 0, 0], [criterion_small, 9, 9]]
    
    D3Scatter(class_num=2, border_points=border_points,
              total_arousal=total_arousal,
              total_dominance=total_dominacne,
              total_valence=total_valence,
              title=scatter_title)


def clear_emotion_3classes(criterion_small, mid_1, mid_2, criterion_big, remote):
    save_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion_3class/"
    rppg_folder = save_dir + "rppg/"
    ppg_folder = save_dir + "ppg/"
    label_folder = save_dir + "label/"
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)

    total_arousal = []
    total_valence = []
    total_dominacne = []
    
    clear_label = []
    clear_ppg = []
    clear_rppg = []
    

def clear_emotion(left_b_point, right_u_point, remote):
    save_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
    rppg_folder = save_dir + "rppg/"
    ppg_folder = save_dir + "ppg/"
    label_folder = save_dir + "label/"
    
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)
    
    total_arousal = []
    total_valence = []
    total_dominacne = []
    
    clear_label = []
    clear_ppg = []
    clear_rppg = []

    h_num = c_num = a_num = s_num = 0
    
    for subject_no in subject_list:
        if remote:
            deap_dataset = deap_rppg_loader(verbose=False)
        else:
            deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
        
        # print(deap_dataset["labels"])
        
        for i in range(40):
            # print(deap_dataset["labels"][i, 0])
            total_arousal.append(deap_dataset["labels"][i, 1])
            total_valence.append(deap_dataset["labels"][i, 0])
            total_dominacne.append(deap_dataset["labels"][i, 2])
            
            # happy
            if deap_dataset["labels"][i, 1] >= right_u_point[1] and deap_dataset["labels"][i, 0] >= right_u_point[0]:
                clear_label.append(0)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
                h_num += 1
            
            # calm
            if deap_dataset["labels"][i, 1] <= left_b_point[1] and deap_dataset["labels"][i, 0] >= right_u_point[0]:
                clear_label.append(1)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
                c_num += 1
                
            # anger
            if deap_dataset["labels"][i, 1] >= right_u_point[1] and deap_dataset["labels"][i, 0] <= left_b_point[0]:
                clear_label.append(2)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
                a_num += 1
                
            # sadness
            if deap_dataset["labels"][i, 1] <= left_b_point[1] and deap_dataset["labels"][i, 0] <= left_b_point[0]:
                clear_label.append(3)
                if remote:
                    clear_rppg.append(deap_dataset["data"][i, 38, :])
                else:
                    clear_ppg.append(deap_dataset["data"][i, 38, :])
                s_num += 1
    
    save_name = (f"{int(left_b_point[0]*10)}_{int(left_b_point[1]*10)}_{int(right_u_point[0]*10)}_{int(right_u_point[1]*10)}.csv")
    clear_label_df = pd.DataFrame(clear_label)
    clear_label_df.to_csv(label_folder + save_name)
    
    if remote:
        clear_rppg_df = pd.DataFrame(clear_ppg)
        clear_rppg_df.to_csv(rppg_folder + save_name)
    else:
        clear_ppg_df = pd.DataFrame(clear_ppg)
        clear_ppg_df.to_csv(ppg_folder + save_name)
    
    center_x = int(right_u_point[0] - left_b_point[0])
    center_y = int(right_u_point[1] - left_b_point[1])
    margin = int(center_x - right_u_point[0])
    
    scatter_title = (f"DEAP dataset label distribution\n"
                     f"Center: ({center_x}, {center_y}), Margin: {margin}\n"
                     f"RED: Happy-{h_num}, GREEN: Calm-{c_num}, MAGENTA: Anger-{a_num}, BLUE: Sadness-{s_num}")
    
    border_points = {}
    border_points["happy"] = [[right_u_point[0], right_u_point[1], 0], [9, 9, 9]]
    border_points["anger"] = [[0, right_u_point[1], 0], [left_b_point[0], 9, 9]]
    border_points["sadness"] = [[0, 0, 0], [left_b_point[0], left_b_point[1], 9]]
    border_points["calm"] = [[right_u_point[0], 0, 0], [9, left_b_point[1], 9]]
    
    D3Scatter(class_num=4, border_points=border_points,
              total_arousal=total_arousal,
              total_dominance=total_dominacne,
              total_valence=total_valence,
              title=scatter_title)

def split_4class(features, label, test_size, random_state):
    happy_label = []
    calm_label = []
    anger_label = []
    sadness_label = []
    
    happy_ppg = []
    calm_ppg = []
    anger_ppg = []
    sadness_ppg = []

    for i, value in enumerate(label.loc[:, '0']):
        if value == 0:
            happy_ppg.append(features.loc[i])
            happy_label.append(value)
        elif value == 1:
            calm_ppg.append(features.loc[i])
            calm_label.append(value)
        elif value == 2:
            anger_ppg.append(features.loc[i])
            anger_label.append(value)
        else:
            sadness_ppg.append(features.loc[i])
            sadness_label.append(value)
    
    save_columns = features.columns
    
    happy_ppg = pd.DataFrame(np.array(happy_ppg)).reset_index(drop=True)
    calm_ppg = pd.DataFrame(np.array(calm_ppg)).reset_index(drop=True)
    anger_ppg = pd.DataFrame(np.array(anger_ppg)).reset_index(drop=True)
    sadness_ppg = pd.DataFrame(np.array(sadness_ppg)).reset_index(drop=True)
    
    happy_label = pd.Series(np.array(happy_label).reshape(-1,)).reset_index(drop=True)
    calm_label = pd.Series(np.array(calm_label).reshape(-1,)).reset_index(drop=True)
    anger_label = pd.Series(np.array(anger_label).reshape(-1,)).reset_index(drop=True)
    sadness_label = pd.Series(np.array(sadness_label).reshape(-1,)).reset_index(drop=True)
    
    x_train_happy, x_test_happy, y_train_happy, y_test_happy = train_test_split(happy_ppg, happy_label, test_size=test_size, random_state=random_state)
    x_train_calm, x_test_calm, y_train_calm, y_test_calm = train_test_split(calm_ppg, calm_label, test_size=test_size, random_state=random_state)
    x_train_anger, x_test_anger, y_train_anger, y_test_anger = train_test_split(anger_ppg, anger_label, test_size=test_size, random_state=random_state)
    x_train_sadness, x_test_sadness, y_train_sadness, y_test_sadness = train_test_split(sadness_ppg, sadness_label, test_size=test_size, random_state=random_state)
    
    X_train = pd.concat([x_train_happy, x_train_calm, x_train_anger, x_train_sadness], axis=0)
    X_test = pd.concat([x_test_happy, x_test_calm, x_test_anger, x_test_sadness], axis=0)
    y_train = pd.concat([y_train_happy, y_train_calm, y_train_anger, y_train_sadness], axis=0)
    y_test = pd.concat([y_test_happy, y_test_calm, y_test_anger, y_test_sadness], axis=0)
    
    X_train.columns = save_columns
    X_test.columns = save_columns
    y_train.columns = [0]
    y_test.columns = [0]
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, (len(y_test_happy), len(y_test_calm), len(y_test_anger), len(y_test_sadness))

def split_3class():
    positive_label = []
    neutral_label = []
    negative_label = []
    
    positive_ppg = []
    neutral_label = []
    negative_ppg = []
    
    for i, value in enumerate(label.loc[:, '0']):
        if value == 0:
            positive_ppg.append(features.loc[i])
            positive_label.append(value)
        elif value == 1:
            neutral_ppg.append(features.loc[i])
            neutral_label.append(value)
        else:
            negative_ppg.append(features.loc[i])
            neutral_label.append(value)
    
    save_columns = features.columns
    
    positive_ppg = pd.DataFrame(np.array(positive_ppg)).reset_index(drop=True)
    neutral_ppg = pd.DataFrame(np.array(neutral_ppg)).reset_index(drop=True)
    negative_ppg = pd.DataFrame(np.array(negative_ppg)).reset_index(drop=True)

    positive_label = pd.Series(np.array(positive_label).reshape(-1,)).reset_index(drop=True)
    neutral_label = pd.Series(np.array(neutral_label).reshape(-1,)).reset_index(drop=True)
    negative_label = pd.Series(np.array(negative_label).reshape(-1,)).reset_index(drop=True)

    x_train_positive, x_test_positive, y_train_positive, y_test_positive = train_test_split(positive_ppg, positive_label, test_size=test_size, random_state=random_state)
    x_train_neutral, x_test_neutral, y_train_neutral, y_test_neutral = train_test_split(neutral_ppg, negative_label, test_size=test_size, random_state=random_state)
    x_train_negative, x_test_negative, y_train_negative, y_test_negative = train_test_split(negative_ppg, negative_label, test_size=test_size, random_state=random_state)

    X_train = pd.concat([x_train_positive, x_train_neutral, x_train_negative], axis=0)
    X_test = pd.concat([x_test_positive, x_test_neutral, x_test_negative], axis=0)
    y_train = pd.concat([y_train_positive, y_train_neutral, y_train_negative], axis=0)
    y_test = pd.concat([y_test_positive, y_test_neutral, y_test_negative], axis=0)
    
    X_train.columns = save_columns
    X_test.columns = save_columns
    y_train.columns = [0]
    y_test.columns = [0]
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, (len(y_test_positive), len(y_test_neutral), len(y_test_negative))

def split_2class(features, label, test_size, random_state):
    positive_label = []
    negative_label = []
    
    positive_ppg = []
    negative_ppg = []
    
    for i, value in enumerate(label.loc[:, '0']):
        if value == 0:
            positive_ppg.append(features.loc[i])
            positive_label.append(value)
        elif value == 1:
            negative_ppg.append(features.loc[i])
            negative_label.append(value)
    
    save_columns = features.columns
    
    positive_ppg = pd.DataFrame(np.array(positive_ppg)).reset_index(drop=True)
    negative_ppg = pd.DataFrame(np.array(negative_ppg)).reset_index(drop=True)

    positive_label = pd.Series(np.array(positive_label).reshape(-1,)).reset_index(drop=True)
    negative_label = pd.Series(np.array(negative_label).reshape(-1,)).reset_index(drop=True)

    x_train_positive, x_test_positive, y_train_positive, y_test_positive = train_test_split(positive_ppg, positive_label, test_size=test_size, random_state=random_state)
    x_train_negative, x_test_negative, y_train_negative, y_test_negative = train_test_split(negative_ppg, negative_label, test_size=test_size, random_state=random_state)

    X_train = pd.concat([x_train_positive, x_train_negative], axis=0)
    X_test = pd.concat([x_test_positive, x_test_negative], axis=0)
    y_train = pd.concat([y_train_positive, y_train_negative], axis=0)
    y_test = pd.concat([y_test_positive, y_test_negative], axis=0)
    
    X_train.columns = save_columns
    X_test.columns = save_columns
    y_train.columns = [0]
    y_test.columns = [0]
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, (len(y_test_positive), len(y_test_negative))


def splitbylabel(features, label, test_size, num_class, random_state):
    if num_class == 4:
        x_train, x_test, y_train, y_test, dataset_size = split_4class(features, label, test_size, random_state)
        return x_train, x_test, y_train, y_test, dataset_size
        
    if num_class == 3:
        x_train, x_test, y_train, y_test, dataset_size = split_3class(features, label, test_size, random_state)
        return x_train, x_test, y_train, y_test, dataset_size
    
    if num_class == 2:
        x_train, x_test, y_train, y_test, dataset_size = split_2class(features, label, test_size, random_state)
        return x_train, x_test, y_train, y_test, dataset_size

def cal_precision(y_pred, y_test, label_class):
    
    tp = 0
    fp = 0
    
    for i in range(len(y_test)):
        if y_pred[i] == label_class:
            if y_test[i] == label_class:
                tp += 1
            else:
                fp += 1
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)

def cal_recall(y_pred, y_test, label_class):
    
    tp = 0
    fn = 0

    y_test = y_test.to_list()
    
    for i in range(len(y_test)):
        if y_test[i] == label_class:
            if y_pred[i] == label_class:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def classnum4performance(y_pred, y_test):
    happy_precision = round(cal_precision(y_pred, y_test, 0), 2)
    calm_precision = round(cal_precision(y_pred, y_test, 1), 2)
    anger_precision = round(cal_precision(y_pred, y_test, 2), 2)
    sadness_precision = round(cal_precision(y_pred, y_test, 3), 2)
    
    happy_recall = round(cal_recall(y_pred, y_test, 0), 2)
    calm_recall = round(cal_recall(y_pred, y_test, 1), 2)
    anger_recall = round(cal_recall(y_pred, y_test, 2), 2)
    sadness_recall = round(cal_recall(y_pred, y_test, 3), 2)

    return (happy_precision, happy_recall), (calm_precision, calm_recall), (anger_precision, anger_recall), (sadness_precision, sadness_recall)

    
def classnum3performance(y_pred, y_test):
    positive_precision = round(cal_precision(y_pred, y_test, 0), 2)
    neutral_precision = round(cal_precision(y_pred, y_test, 1), 2)
    negative_precision = round(cal_precision(y_pred, y_test, 2), 2)
    
    positive_recall = round(cal_recall(y_pred, y_test, 0), 2)
    neutral_precision = round(cal_recall(y_pred, y_test, 1), 2)
    negative_recall = round(cal_recall(y_pred, y_test, 2), 2)
    
    return (positive_precision, positive_recall), (neutral_precision, neutral_recall), (negative_precision, negative_recall)

def classnum2performance(y_pred, y_test):
    positive_precision = round(cal_precision(y_pred, y_test, 0), 2)
    negative_precision = round(cal_precision(y_pred, y_test, 1), 2)
    
    positive_recall = round(cal_recall(y_pred, y_test, 0), 2)
    negative_recall = round(cal_recall(y_pred, y_test, 1), 2)
    
    return (positive_precision, positive_recall), (negative_precision, negative_recall)


class classification_1by1:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.random_state = 42
        self.num_class = 0
        if isinstance(self.features, dict):
            self.iter = features.keys()[-1]
            if self.features.keys()[-1] == 4:
                self.class4()
                self.num_class = 4
            else:
                self.class3()
                self.num_class = 3
        else:
            self.class2()
            self.num_class = 2
        
    def class4(self):
        self.X_train_happy, self.X_test_happy, self.y_train_happy, self.y_test_happy, self.dataset_size_happy = splitbylabel(self.features[0], self.label, 0.2, 4, 42)
        self.X_train_calm, self.X_test_calm, self.y_train_calm, self.y_test_calm, self.dataset_size_calm = splitbylabel(self.features[1], self.label, 0.2, 4, 42)
        self.X_train_anger, self.X_test_anger, self.y_train_anger, self.y_test_anger, self.dataset_size_anger = splitbylabel(self.features[2], self.label, 0.2, 4, 42)
        self.X_train_sadness, self.X_test_sadness, self.y_train_sadness, self.y_test_sadness, self.dataset_size_sadness = splitbylabel(self.features[3], self.label, 0.2, 4, 42)
        self.X_train_mean, self.X_test_mean, self.y_train_mean, self.y_test_mean, self.dataset_size_mean = splitbylabel(self.features[4], self.label, 0.2, 4, 42)
        
    def class3(self):
        self.X_train_positive, self.X_test_positive, self.y_train_positive, self.y_test_positive, self.dataset_size_positive = splitbylabel(self.features[0], self.label, 0.2, 3, 42)
        self.X_train_neutral, self.X_test_neutral, self.y_train_neutral, self.y_test_neutral, self.dataset_size_neutral = splitbylabel(self.features[1], self.label, 0.2, 3, 42)
        self.X_train_negative, self.X_test_negative, self.y_train_negative, self.y_test_negative, self.dataset_size_negative = splitbylabel(self.features[2], self.label, 0.2, 3, 42)

    def class2(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.dataset_size = splitbylabel(self.features, self.label, 0.2, 3, 42)

    def xgboost(self):
        if self.num_class == 4:
            xgb_happy = xgb(n_estimators=100, random_state=self.random_state)
            xgb_calm = xgb(n_estimators=100, random_state=self.random_state)
            xgb_anger = xgb(n_estimators=100, random_state=self.random_state)
            xgb_sadness = xgb(n_estimators=100, random_state=self.random_state)
            xgb_mean = xgb(n_estimators=100, random_state=self.random_state)
            
            xgb_happy.fit(self.X_train_happy, self.y_train_happy)
            xgb_calm.fit(self.X_train_calm, self.y_train_calm)
            xgb_happy.fit(self.X_train_happy, self.y_train_happy)
            xgb_happy.fit(self.X_train_happy, self.y_train_happy)

    def random_forst(self):
        

    def svm(self):
        return 0

    def onevsrest(self):
        return 0
    
    def oneveone(self):
        return 0
    
    def outputcode(self):
        return 0


def SHAP_ranking(feature_list, shap_value):
    output = []
    if len(shap_value.shape) >= 3:
        temp = []
        for i in range(shap_value.shape[-1]):
            sorted_index = np.argsort(np.mean(np.abs(shap_value[:, :, i]), axis=0)).tolist()
            temp.append(sorted_index)
            output.append([feature_list[sorted_index.index(i)] for i in range(len(sorted_index))])
        
        temp = np.array(temp).T
        temp = np.argsort(np.mean(temp, axis=1)).tolist()
        output.append([feature_list[temp.index(i)] for i in range(len(temp))])
        output = np.array(output)
    else:
        sorted_index = np.argsort(np.mean(np.abs(shap_value[:, :, i]), axis=0)).tolist()
        output = np.array([feature_list[sorted_index.index(i)] for i in range(len(sorted_index))])
            
    return output


def SHAP(features, label, num_class, parameter_num):
    
    """
    input:
        features: HRV features for classification
        label: Clear label (relabled : point -> class)
        num_class: The number of classes
        parameter_num: Paramter number for performance test
                       These
    
    output:
        shap_values: shap values for parameter ranking
                     Due to this output, good10 will be deprecated.
    """
    
    plt.rcParams['font.size'] = 15
    
    X_train, X_test, y_train, y_test, dataset_size = splitbylabel(features, label, test_size=0.2, num_class=num_class, random_state=42)
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    rankings = np.zeros((shap_values.values.shape[1], shap_values.values.shape[-1]))
    
    # good10 (any) -> index of parameters good for classification (parameter_num)
    
    if len(shap_values.values.shape)>=3:  # 4, 3 classes
        for i in range(shap_values.values.shape[-1]):
            shap.plots.beeswarm(shap_values[:, :, i], max_display=12)
        
        for i in range(shap_values.values.shape[-1]):
            mean_abs_shap_values = np.mean(np.abs(shap_values.values[:, :, i]), axis=0)
    
            sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
            for rank, index in enumerate(sorted_indices):
                rankings[index, class_index] = rank
        average_rankings = np.mean(rankings, axis=1)
        sorted_indices_by_avg_rank = np.argsort(average_rankings)
        sorted_parameters_by_avg_rank = [X_test.columns[i] for i in sorted_indices_by_avg_rank]
    else:
        shap.plots.beeswarm(shap_values, max_display=12)
        sorted_indices = np.argsort(np.mean(np.abs(shap_values.values), axis=0))
        sorted_parameters_by_avg_rank = [X_test.columns[i]]
        
    good10 = sorted_parameters_by_avg_rank[0:10]

    
    return shap_values.values


def test(base_dir, file_name, num_class, parameter_num, remote=True):
    if remote:
        print("not implemented")
        sys.exit()
        
    else:
        ppg, label, sampling_rate = ppg_loader4deap(base_dir, file_name)
        # print(f"\n\nppg.shape: {ppg.shape}, label.shape: {label.shape}\n\n")
        
        features = pd.DataFrame([0])
        for i in range(ppg.shape[0]):
            signals, info = nk.ppg_process(ppg.loc[i, :], sampling_rate=sampling_rate)
            ppg_raw = signals["PPG_Raw"]
            ppg_clean = signals["PPG_Clean"]
            ppg_rate = signals["PPG_Clean"]
            ppg_quality = signals["PPG_Quality"]
            ppg_peaks = signals["PPG_Peaks"]
            
            # custom_scale = nk.expspace(10, 1000, 20, base=2)
            result = nk.hrv(ppg_peaks, sampling_rate=sampling_rate, show=False)
            result = result.dropna(axis=1)
            
            if i == 0:
                features = pd.DataFrame(result)
            else:
                temp = pd.DataFrame(result)
                features = pd.concat([features, temp], axis=0, join='outer')
        
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.reset_index(drop=True)
        label = label.reset_index(drop=True)
        nan_rows = features[features.isna().any(axis=1)].index
        
        features = features.dropna(axis=0)
        features = features.reset_index(drop=True)
        label = label.drop(nan_rows)
        label = label.reset_index(drop=True)
    
    performance_y_label = ["Precision", "Recall"]
    
    feature_list = features.columns
    
    if class_num == 4:
        shap_values = SHAP(features, label, num_class, parameter_num=parameter_num)
        
        sel_parameters = SHAP_ranking(feature_list, shap_values)
        
        happy_input = features.loc[:, sel_parameters[0, :parameter_num]]
        calm_input = features.loc[:, sel_parameters[1, :parameter_num]]
        anger_input = features.loc[:, sel_parameters[2, :parameter_num]]
        sadness_input = features.loc[:, sel_parameters[3, :parameter_num]]
        mean_input = features.loc[:, sel_parameters[4, :parameter_num]]
        
        onebyone_input = {}
        onebyone_input[0] = happy_input
        onebyone_input[1] = calm_input
        onebyone_input[2] = anger_input
        onebyone_input[3] = sadness_input
        onebyone_input[4] = mean_input
        
        class4_1by1 = classification_1by1(onebyone_input, label)
        
        y_pred_or, y_test_or, accuracy_or, happy_or, calm_or, anger_or, sadness_or, test_size_or = class4_1by1.onevsrest()
        y_pred_oo, y_test_oo, accuracy_oo, happy_oo, calm_oo, anger_oo, sadness_oo, test_size_oo = calss4_1by1.onevsone()
        y_pred_oc, y_test_oc, accuracy_oc, happy_oc, calm_oc, anger_oc, sadness_oc, test_size_oc = class4_1by1.outputcode()
        
        cm_or = confusion_matrix(y_test_or, y_pred_or)
        cm_oo = confusion_matrix(y_test_oo, y_pred_oo)
        cm_oc = confusion_matrix(y_test_oc, y_pred_oc)
        
        emotions = ["happy", "calm", "anger", "sadness"]
        
        xg_total = [happy_xg, calm_xg, anger_xg, sadness_xg]
        svm_total = [happy_sv, calm_sv, anger_sv, sadness_sv]
        rf_total = [happy, calm, anger, sadness]
        
        xg_title = (f"Parameters\n{good10[0:5]}\n{good10[5:]}\n\nXGBoost result"
                    f"\ntest size Happy: {test_size_xg[0]} Calm: {test_size_xg[1]} Anger: {test_size_xg[2]} Sadness: {test_size_xg[3]}")
        
        sv_title = (f"Paramters")
        
        rf_title = (f"Random Forest result"
                    f"\ntest size Happy: {test_size[0]} Calm: {test_size[1]} Anger: {test_size[2]} Sadness: {test_size[3]}")
        
        performance_array = np.array([[happy, calm, anger, sadness], [happy_xg, calm_xg, anger_xg, sadness_xg]])
        
    elif class_num == 3:
        shap_values = SHAP(features, label, num_class=num_class, parameter_num=parameter_num)
        features = features.loc[:, good10]
        y_pred, y_test, accuracy, positive, neutral, negative, test_size = rf(features, label, num_class=num_class)
    
        cm = confusion_matrix(y_test, y_pred)
        cm_xg = confision_matrix(y_test_xg, y_pred_xg)
        
        emotions = ["positive", "neutral", "negative"]
        
        xg_total = [positive_xg, neutral_xg, negative_xg]
        rf_total = [positive, neutral, negative]
        
        xg_title = (f"Parameters\n{good10[0:5]}\n{good10[5:]}\n\nXGBoost result"
                    f"\ntest size Positive: {test_size_xg[0]} Neutral: {test_size_xg[1]} Negative: {test_size_xg[2]}")
        
        rf_title = (f"Random Forest result"
                    f"\ntest size Positive: {test_size[0]} Neutral: {test_size[1]} Negative: {test_size[2]}")
        
        performance_array = np.array([[positive, neutral, negative], [positive_xg, neutral_xg, negative_xg]])
    
    else:
        shap_values = SHAP(features, label, num_class=num_class, parameter_num=parameter_num)
        features = features.loc[:, good10]
        y_pred, y_test, accuracy, positive, negative, test_size = rf(features, label, num_class=num_class)
    
        cm = confusion_matrix(y_test, y_pred)
        cm_xg = confusion_matrix(y_test_xg, y_pred_xg)
        
        emotions = ["positive", "negative"]
        
        xg_total = [positive_xg, negative_xg]
        rf_total = [positive, negative]
        
        
        xg_title = (f"Paramters\n{good10[0:5]}\n{good10[5:]}\n\nXGBoost result"
                    f"\ntest size Positive: {test_size_xg[0]} Negative: {test_size_xg[1]}")
        
        rf_title = (f"Random Forest result"
                    f"\ntest size Positive: {test_size[0]} Negative: {test_size[1]}")
    
        performance_array = np.array([[positive, negative], [positive_xg, negative_xg]])
    
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 20))
    sns.heatmap(cm, ax=ax[0, 0], fmt='d', cmap='Blues', annot=True, xticklabels=emotions, yticklabels=emotions)
    sns.heatmap(cm_xg, ax=ax[1, 0], fmt='d', cmap='Blues', annot=True, xticklabels=emotions, yticklabels=emotions)
    
    ax[1, 0].set_title(xg_title)
    
    ax[0, 0].set_title(rf_title)
    
    c1 = ax[0, 1].imshow(performance_array[:, :, 0], cmap='Oranges')
    
    c2 = ax[1, 1].imshow(performance_array[:, :, 1], cmap='Oranges')
    
    ax[0, 1].set_title("Random Forest Performance")
    fig.colorbar(c1, ax=ax[0, 1])
    
    ax[1, 1].set_title("XGBoost Performance")
    fig.colorbar(c2, ax=ax[1, 1])
    
    for i in range((len(emotions))):
        for j in range(len(performance_y_label)):
            ax[0, 1].text(i, j, f"{xg_total[i][j]}", ha='center', va='center', color='black')
            ax[1, 1].text(i, j, f"{rf_total[i][j]}", ha='center', va='center', color='black')
    
    ax[0, 1].set_xticks(np.arange(len(emotions)))
    ax[0, 1].set_xticklabels(emotions)
    ax[0, 1].set_yticks(np.arange(len(performance_y_label)))
    ax[0, 1].set_yticklabels(performance_y_label)
    
    ax[1, 1].set_xticks(np.arange(len(emotions)))
    ax[1, 1].set_xticklabels(emotions)
    ax[1, 1].set_yticks(np.arange(len(performance_y_label)))
    ax[1, 1].set_yticklabels(performance_y_label)
    
    ax[1, 0].set_xlabel('Predicted Label')
    ax[0, 0].set_xlabel("Predicted label")
    ax[1, 0].set_ylabel('True Label')
    ax[0, 0].set_ylabel("True Label")
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
    return y_pred, y_test, accuracy
    

def deap_classification(class_num, remote=False, emotionless=False, verbose=False,
                        parameter_num=10):
    
    """
    criterion: Dataset class border list
               If 2 classes, element length: 2
               -> [negative border, positive border]
               
               If 3 classes, element length: 4
               -> [negative border, neutral start border, neutral end border, positive border]
               
               If 4 classes and emotionless is False, element length: 2
               (In valence(x axis) - arousal(y axis) plane)
               -> [Left below point, Right upper point]
               Each point has two values [x, y]
               
               If 4 classes and emotionless is True, element length: 4
               (In valence(x axis) - arousal(y axis) plane)
               -> [Left below point, Right upper point,
                   Neutral left upper point, Neutral right below point]
               Each point has two values [x, y]
               
               
                Russell's 2D emotion model
               
                         Arousal (Y axis)
                            ^
                            |
                            |
                  Anger     |     HAPPY
                            |
                            |
             ------------------------------>  Valence (X axis)
                            |
                            |
                 Sadness    |     Calm
                            |
                            |
                            |
                
                Reference:
                https://dl.acm.org/doi/pdf/10.1145/3297156.3297177
                Emotion Classification Using EEG siganls
    """
    criterion = []
    
    # Valence minimum value and maximum value
    num_min = 2.0
    num_max = 7.0

    # Dataset cutting criterion decision (0 ~ 8)
    if class_num == 2:
        print("Positive / Negative classification start")
        for i in range(int((num_max - num_min + 1)*2)):
            offset = 0
            while True:
                criterion.append([num_min+i*0.5, num_min+i*0.5+offset])
                offset += 0.5
                if 1+i*0.5+offset > num_max:
                    break
        if verbose:
            for border in criterion:
                plt.figure(figsize=(15,15))
                plt.axvspan(0, border[0], color='blue', alpha=1.0, label="Negative")
                plt.axvspan(border[1], 8, color='orange', alpha=1.0, label="Positive")
                plt.legend()
                plt.xlabel("Valence")
                plt.ylabel("Arousal")
                plt.xlim(0, 10)
                plt.ylim(0, 10)
                plt.show()
        
    if class_num == 3:
        print("Positive / Neutral / Negative classification start")
        mid_scale = 1
        for _ in range(int((num_max-num_min-1)//0.5)):
            offset = 0
            while True:
                bottom = 1
                while True:
                    b = bottom + offset
                    c = b + mid_scale
                    d = c + offset
                    if d > num_max:
                        break
                    criterion.append([bottom, b, c, d])
                    bottom += 0.5
                offset += 0.5
                if offset > (num_max-num_min-1)//2:
                    break
            mid_scale += 0.5
            
        if verbose:
            for border in criterion:
                plt.figure(figsize=(15, 15))
                plt.axvspan(0, border[0], color='blue', alpha=1.0, label="Negative")
                plt.axvspan(border[1], border[2], color='yellow', alpha=1.0, label="Neutral")
                plt.axvspan(border[3], 8, color='orange', alpha=1.0, label="Positive")
                plt.legend()
                plt.xlabel("Valence")
                plt.ylabel("Arousal")
                plt.xlim(0, 10)
                plt.ylim(0, 10)
                plt.show()
            
    if class_num == 4 and not emotionless:
        print("Happy / Anger / Sadness / Calm classification start")
        criteria_list = [1.5 + 0.5*i for i in range(11)]

        for x in criteria_list:
            for y in criteria_list:
                offset = 0
                while True:
                    x_lb = x-offset
                    y_lb = y-offset
                    x_ru = x+offset
                    y_ru = y+offset
                    if x_lb < num_min+0.5 or x_ru > num_max+0.5 or y_lb < num_min+0.5 or y_ru > num_max+0.5:
                        break
                    criterion.append([[x_lb, y_lb], [x_ru, y_ru]])
                    offset += 0.5
                
        if verbose:
            for border in criterion:
                fig, ax = plt.subplots(figsize=(15, 15))
                rect1 = patches.Rectangle((0, 0), border[0][0], border[0][1], edgecolor='b', facecolor='b')  # sadness
                rect2 = patches.Rectangle((0, border[1][1]), border[0][0], 8-border[1][1], edgecolor='m', facecolor='m')  # anger
                rect3 = patches.Rectangle((border[1][0], border[1][1]), 8-border[1][0], 8-border[1][1], edgecolor='r', facecolor='r')  # happy
                rect4 = patches.Rectangle((border[1][0], 0), 8-border[1][0], border[0][1], edgecolor='g', facecolor='g')  # calm
                ax.text(border[0][0]//2, border[0][1]//2, 'sadness', color='black', fontsize=30, ha='center', va='center')
                ax.text(border[0][0]//2, border[1][1]+(8-border[1][1])//2, 'anger', color='black', fontsize=30, ha='center', va='center')
                ax.text(border[1][0]+(8-border[1][0])//2, border[1][1]+(8-border[1][1])//2, 'happy', color='black', fontsize=30, ha='center', va='center')
                ax.text(border[1][0]+(8-border[1][0])//2, border[0][1]//2, 'calm', color='black', fontsize=30, ha='center', va='center')
                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                ax.add_patch(rect4)
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.set_xlabel("Valence")
                ax.set_ylabel("Arousal")
                plt.legend()
                plt.show()
    
    if class_num == 4 and emotionless:
        print("Happy / Anger / Sadness / Neutral classification start")
        criteria_list = [1.5 + 0.5*i for i in range(11)]
        for x in criteria_list:
            for y in criteria_list:
                pass

    print("criterion: ", len(criterion))
    
    total_acc = []
    total_pred = []
    total_test = []
    iter_num = len(criterion)
    print("start")
    
    for i, border in enumerate(criterion):
        file_name = ""
        for value in border:
            append_str = '_' + str(value)
            file_name += append_str
        file_name = file_name + ".csv"
        
        if class_num == 2:
            clear_emotion_binary(border[0], border[1], remote=remote)
            file_name = f"{int(border[0]*10)}_{int(border[1]*10)}.csv"
            base_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion_binary/"
        if class_num == 3:
            clear_emotion_3classes(criterion_small, mid_1, mid_2, criterion_big, remote)
            file_name = f"{int(border[0]*10)}_{int(border[1]*10)}_{int(border[2]*10)}_{int(border[3]*10)}.csv"
            base_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion_3class/"
        if class_num == 4 and not emotionless:
            clear_emotion(border[0], border[1], remote=False)
            file_name = f"{int(border[0][0]*10)}_{int(border[0][1]*10)}_{int(border[1][0]*10)}_{int(border[1][1]*10)}.csv"
            base_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
        
        y_pred, y_test, accuracy = test(base_dir, file_name, num_class=class_num,
                                        parameter_num=parameter_num, remote=remote)
        
        total_acc.append(accuracy)
        total_pred.append(y_pred)
        total_test.append(y_test)
        print(f"{round((i+1)/iter_num*100,2)}%")
    
    x_ticks = np.linspace(0, len(accuracy)-1, len(accuracy))
    combination_str = [str(item1) + ' ' + str(item2) for item1, item2 in combinations_float]
    plt.figure(figsize=(15,7))
    plt.plot(accuracy)
    plt.plot(total_test_std)
    plt.xticks(x_ticks, combinations_str)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Combination")
    plt.title("Accuracy & Test dataset std over combination")
    plt.show()
    
# seinor dataset label order
# happy > neutral > anxiety > embarrassment > heartache > sadness > anger

if __name__ == "__main__":
    class_num = 2
    remote = False
    verbose = False
    parameter_num = 10
    deap_classification(class_num=class_num, remote=remote, verbose=verbose, parameter_num=parameter_num)
    
    