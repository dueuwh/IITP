import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import shap
import xgboost as xgb
from sklearn.metrics import confusion_matrix
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

def standardization():
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


def ppg_loader4deap(file_name):
    base_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
    ppg = pd.read_csv(base_dir + "ppg/" + file_name, index_col=0)
    label = pd.read_csv(base_dir + "label/" + file_name, index_col=0)
    
    # The sampling rate of DEAP dataset is 128
    
    return ppg, label, 128.0


def rppg_loader4deap(file_name):
    pass

 

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
        cube(ax, [0, 0, 0], [8, 8, criterion_small], 'g')
        cube(ax, [0, 0, criterion_big], [8, 8, 8], 'r')
    
    elif class_num == 3:
        cube(ax, [0, 0, 0], [8, 8, criterion_small], 'r')
        cube(ax, [0, 0, mid1], [8, 8, mid2], 'g')
        cube(ax, [0, 0, criterion_big], [8, 8, 8], 'b')
    
    elif class_num == 4:
        cube(ax, border_points["happy"][0], border_points["happy"][1], 'r')
        cube(ax, border_points["anger"][0], border_points["anger"][1], 'm')
        cube(ax, border_poitns["sadness"][0], border_poitns["sadness"][1], 'b')
        cube(ax, border_poitns["calm"][0], border_poitns["calm"][1] 'g')
    
    ax.view_init(elev=30, azim=45)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    
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
    
    title=f"DEAP dataset label distribution ({criterion_small}, {criterion_big})\nRED: Positive-{p_num} Negative-{n_num}"
    
    D3Scatter(class_num=2, criterion_small=criterion_small, mid1=None, mid2=None, criterion_big=criterion_big,
              total_arousal=total_arousal,
              total_dominance=total_dominacne,
              total_valence=total_valence,
              title=title)


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
    
    save_name = str(left_b_point[0]) + '_' + str(left_b_point[1]) + '_' + str(right_u_point[0]) + '_' + str(right_u_point[1]) + ".csv"
    clear_label_df = pd.DataFrame(clear_label)
    clear_label_df.to_csv(label_folder + save_name
    
    if remote:
        clear_rppg_df = pd.DataFrame(clear_ppg)
        clear_rppg_df.to_csv(rppg_folder + save_name)
    else:
        clear_ppg_df = pd.DataFrame(clear_ppg)
        clear_ppg_df.to_csv(ppg_folder + save_name)
    
    center_x = int(right_b_point[0] - left_b_point[0])
    center_y = int(right_u_point[1] - left_b_point[1])
    margin = int(center_x - right_b_point[0])
    
    scatter_title = f"DEAP dataset label distribution\n"+
                     "Center: ({center_x}, {center_y}), Margin: {margin}\n"+
                     "RED: Happy-{h_num}, GREEN: Calm-{c_num}, MAGENTA: Anger-{a_num}, BLUE: Sadness-{s_num}"
    
    border_points = {}
    border_points["happy"] = [[right_u_point[0], right_u_point[1], 0], [8, 8, 8]]
    border_points["anger"] = [[0, right_u_point[1], 0], [left_b_point[0], right_u_point[1], 8]]
    border_points["sadness"] = [[left_b_point[0], left_b_point[1], 0], []]
    border_points["calm"] = [[], []]
    
    D3Scatter(class_num=4, border_points,
              total_arousal=total_arousal,
              total_dominance=total_dominacne,
              total_valence=total_valence,
              title=scatter_title)


def splitbylabel(features, label, test_size, random_state):
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

def rf(features, label):
    X_train, X_test, y_train, y_test, dataset_size = splitbylabel(features, label, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    happy_precision = round(cal_precision(y_pred, y_test, 0), 2)
    calm_precision = round(cal_precision(y_pred, y_test, 1), 2)
    anger_precision = round(cal_precision(y_pred, y_test, 2), 2)
    sadness_precision = round(cal_precision(y_pred, y_test, 3), 2)
    
    happy_recall = round(cal_recall(y_pred, y_test, 0), 2)
    calm_recall = round(cal_recall(y_pred, y_test, 1), 2)
    anger_recall = round(cal_recall(y_pred, y_test, 2), 2)
    sadness_recall = round(cal_recall(y_pred, y_test, 3), 2)
    
    return y_pred, X_test, y_test, accuracy, (happy_precision, happy_recall), (calm_precision, calm_recall), (anger_precision, anger_recall), (sadness_precision, sadness_recall), dataset_size

def SHAP(features, label):
    
    plt.rcParams['font.size'] = 20
    
    X_train, X_test, y_train, y_test, dataset_size = splitbylabel(features, label, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    rankings = np.zeros((shap_values.shape[1], shap_values.shape[-1]))
    for i in range(shap_values.shape[-1]):
        shap.plots.beeswarm(shap_values[:, :, i], max_display=12)
    
    for i in range(shap_values.shape[-1]):
        mean_abs_shap_values = np.mean(np.abs(shap_values[:, :, i]), axis=0)

        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        for rank, index in enumerate(sorted_indices):
            rankings[index, class_index] = rank

    average_rankings = np.mean(rankings, axis=1)
    sorted_indices_by_avg_rank = np.argsort(average_rankings)
    sorted_parameters_by_avg_rank = [X_test.columns[i] for i in sorted_indices_by_avg_rank]
    
    good10 = sorted_parameters_by_avg_rank[0:10]

    X_train = X_train.loc[:, good10]
    X_test = X_test.loc[:, good10]

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    happy_precision = round(cal_precision(y_pred, y_test, 0), 2)
    calm_precision = round(cal_precision(y_pred, y_test, 1), 2)
    anger_precision = round(cal_precision(y_pred, y_test, 2), 2)
    sadness_precision = round(cal_precision(y_pred, y_test, 3), 2)
    
    happy_recall = round(cal_recall(y_pred, y_test, 0), 2)
    calm_recall = round(cal_recall(y_pred, y_test, 1), 2)
    anger_recall = round(cal_recall(y_pred, y_test, 2), 2)
    sadness_recall = round(cal_recall(y_pred, y_test, 3), 2)

    # print("shape: ", shap_values.shape)

    # for i in range(shap_values.shape[-1]):
    #     shap.plots.waterfall(shap_values[0, :, i], max_display=10, show=True)
    #     plt.show()
    
    return accuracy, (happy_precision, happy_recall), (calm_precision, calm_recall), (anger_precision, anger_recall), (sadness_precision, sadness_recall), y_pred, y_test, dataset_size, good10


def test(file_name, remote=True):
    if remote:
        ppg, label, sampling_rate = ppg_loader4deap(file_name)
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
        # print(f"\n\nfeatures.shape: {features.shape}, label.shape: {label.shape}\n\n")
        # print(f"\n\nfeatures\n{features}\n\n")
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.reset_index(drop=True)
        label = label.reset_index(drop=True)
        nan_rows = features[features.isna().any(axis=1)].index
        # print(f"\n\nnan_rows:\n{nan_rows}\n\n")
        features = features.dropna(axis=0)
        features = features.reset_index(drop=True)
        label = label.drop(nan_rows)
        label = label.reset_index(drop=True)
        # print(f"\n\nfeatures.shape: {features.shape}, label.shape: {label.shape}\n\n")
        
        # print(f"label:\n{label}")
    else:
        # = deap_rppg_loader(verbose=False)
        pass
        
    xgboost_acc, happy_xg, calm_xg, anger_xg, sadness_xg, y_pred_xg, y_test_xg, test_size_xg, good10 = SHAP(features, label)
    
    features = features.loc[:, good10]
    
    y_pred, X_test, y_test, accuracy, happy, calm, anger, sadness, test_size = rf(features, label)
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm_xg = confusion_matrix(y_test_xg, y_pred_xg)
    
    emotions = ["happy", "calm", "anger", "sadness"]
    
    performance_y_label = ["Precision", "Recall"]
    
    xg_total = [happy_xg, calm_xg, anger_xg, sadness_xg]
    rf_total = [happy, calm, anger, sadness]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 20))
    sns.heatmap(cm, ax=ax[0, 0], fmt='d', cmap='Blues', annot=True, xticklabels=emotions, yticklabels=emotions)
    sns.heatmap(cm_xg, ax=ax[1, 0], fmt='d', cmap='Blues', annot=True, xticklabels=emotions, yticklabels=emotions)
    # ax[0].heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    # ax[1].heatmap(cm_xg, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    ax[1, 0].set_title(f"Parameters\n{good10[0:5]}\n{good10[5:]}\n\nXGBoost result"
                    + f"\ntest size Happy: {test_size_xg[0]} Calm: {test_size_xg[1]} Anger: {test_size_xg[2]} Sadness: {test_size_xg[3]}")
    ax[0, 0].set_title(f"Random Forest result"
                    + f"\ntest size Happy: {test_size[0]} Calm: {test_size[1]} Anger: {test_size[2]} Sadness: {test_size[3]}")
    
    c1 = ax[0, 1].imshow(np.array([[happy[0], calm[0], anger[0], sadness[0]],
                                   [happy[1], calm[1], anger[1], sadness[1]]]),
                         cmap='Oranges')
    
    ax[0, 1].set_title("Random Forest Performance")
    fig.colorbar(c1, ax=ax[0, 1])
    
    c2 = ax[1, 1].imshow(np.array([[happy_xg[0], calm_xg[0], anger_xg[0], sadness_xg[0]],
                                   [happy_xg[1], calm_xg[1], anger_xg[1], sadness_xg[1]]]),
                         cmap='Oranges')
    for i in range((len(emotions))):
        for j in range(len(performance_y_label)):
            ax[0, 1].text(i, j, f"{xg_total[i][j]}", ha='center', va='center', color='black')
            ax[1, 1].text(i, j, f"{rf_total[i][j]}", ha='center', va='center', color='black')
    ax[1, 1].set_title("XGBoost Performance")
    fig.colorbar(c2, ax=ax[1, 1])
    
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
    

def deap_classification(class_num, remote=False, emotionless=False, verbose=False):
    
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
    """
    criterion = []
    
    # Valence minimum value and maximum value
    num_min = 1.5
    num_max = 6.5

    # Dataset cutting criterion decision (0 ~ 8)
    if class_num == 2:
        print("Positive / Negative classification start")
        for i in range(int((num_max - num_min + 1)*2)):
            offset = 0
            while True:
                criterion.append([1+i*0.5, 1+i*0.5+offset])
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
                    if x-offset > num_min and x+offset < num_max and y-offset > num_min and y+offset < num_max:    
                        criterion.append([[x-offset, y-offset], [x+offset,y+offset]])
                    else:
                        break
    
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
            clear_emotion_binary(criterion_small, criterion_big, remote=remote)
        if class_num == 3:
            clear_emotion_3classes(criterion_small, mid_1, mid_2, criterion_big, remote)
        if class_num == 4 and not emotionless:
            clear_emotion(border[0], border[1], large, remote=False)
        
        y_pred, y_test, accuracy = test(file_name, remote=remote)
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
    
    class_num = 4
    remote = False
    deap_classification(class_num=class_num, remote=remote)