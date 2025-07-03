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
from sklearn.metrics import accuracy_score
import shap
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools

def ploting_rppg():
    base_dir = "D:/home/BCML/IITP/data/DEAP/rppg/emma/"
    subject_list = os.listdir(base_dir)
    
    for subject in subject_list:
        folder_list = os.listdir(base_dir+subject)
        for folder in folder_list:
            sample = f"D:/home/BCML/IITP/data/DEAP/rppg/emma/{subject}/{folder}/ppg_omit.txt"
            
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
            
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(200,20))
            plt.plot(crop, marker='o', markersize=6, markerfacecolor='red', markeredgecolor='red')
            plt.title("crop")
            plt.show()


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


def clear_emotion(criterion_small, criterion_big):
    save_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
    rppg_folder = save_dir + "rppg/"
    ppg_folder = save_dir + "ppg/"
    label_folder = save_dir + "label/"
    
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)
    
    total_label = {}
    total_ppg = {}
    
    total_arousal = []
    total_valence = []
    total_dominacne = []
    
    clear_label = []
    clear_ppg = []
    
    # happy = 0, calm = 1, anger = 2, sadness = 3
    
    # criterion_small = 5.0
    # criterion_big = 6.0
    
    clear_3_label = []
    clear_3_ppg = []
    
    negative_board = 3.5
    positive_board = 5.5
    mid_1 = 3.5
    mid_2 = 5.5
    
    p = 0
    m = 0
    n = 0
    
    for subject_no in subject_list:
        deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
        total_ppg[subject_no] = deap_dataset["data"]
        total_label[subject_no] = deap_dataset["labels"]
        
        # print(deap_dataset["labels"])
        
        for i in range(40):
            # print(deap_dataset["labels"][i, 0])
            total_arousal.append(deap_dataset["labels"][i, 1])
            total_valence.append(deap_dataset["labels"][i, 0])
            total_dominacne.append(deap_dataset["labels"][i, 2])
            
            # happy
            if deap_dataset["labels"][i, 1] >= criterion_big and deap_dataset["labels"][i, 0] >= criterion_big:
                clear_label.append(0)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
            
            # calm
            if deap_dataset["labels"][i, 1] <= criterion_small and deap_dataset["labels"][i, 0] >= criterion_big:
                clear_label.append(1)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                
            # anger
            if deap_dataset["labels"][i, 1] >= criterion_big and deap_dataset["labels"][i, 0] <= criterion_small:
                clear_label.append(2)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                
            # sadness
            if deap_dataset["labels"][i, 1] <= criterion_small and deap_dataset["labels"][i, 0] <= criterion_small:
                clear_label.append(3)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                
            # positive
            if deap_dataset["labels"][i, 0] > positive_board:
                clear_3_label.append(0)
                clear_3_ppg.append(deap_dataset["data"][i, 38, :])
                p += 1
            
            # neutral
            if mid_1 <= deap_dataset["labels"][i, 0] <= mid_2:
                clear_3_label.append(1)
                clear_3_ppg.append(deap_dataset["data"][i, 38, :])
                m += 1
            
            # negative
            if deap_dataset["labels"][i, 0] < negative_board:
                clear_3_label.append(2)
                clear_3_ppg.append(deap_dataset["data"][i, 38, :])
                n += 1
    
    # print(f"P:{p}, M:{m}, N:{n}")
    
    clear_label_df = pd.DataFrame(clear_label)
    clear_ppg_df = pd.DataFrame(clear_ppg)
    
    clear_3_label_df = pd.DataFrame(np.array(clear_3_label).T)
    clear_3_ppg_df = pd.DataFrame(np.array(clear_3_ppg).T)
    
    clear_label_df.to_csv(label_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    clear_ppg_df.to_csv(ppg_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    
    clear_3_label_df.to_csv(label_folder + f"{int(negative_board*10)}_{int(mid_1*10)}_{int(mid_2*10)}_{int(positive_board*10)}_3class.csv")
    clear_3_ppg_df.to_csv(ppg_folder + f"{int(negative_board*10)}_{int(mid_1*10)}_{int(mid_2*10)}_{int(positive_board*10)}_3class.csv")
    
    plt.rcParams['font.size'] = 20

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.zeros((len(total_arousal), 3))
    colors[:, 0] = minmaxscalar(total_arousal)
    colors[:, 1] = minmaxscalar(total_dominacne)
    colors[:, 2] = minmaxscalar(total_valence)
    
    norm_x = Normalize(vmin=min(total_arousal), vmax=max(total_arousal))
    norm_y = Normalize(vmin=min(total_dominacne), vmax=max(total_dominacne))
    norm_z = Normalize(vmin=min(total_valence), vmax=max(total_valence))
    
    cmap_x = plt.cm.Reds
    cmap_y = plt.cm.Blues
    cmap_z = plt.cm.Greens
    
    axins_x = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.60, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_y = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.65, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_z = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.70, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    
    cbar_x = fig.colorbar(plt.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_x)
    cbar_y = fig.colorbar(plt.cm.ScalarMappable(norm=norm_y, cmap=cmap_y), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_y)
    cbar_z = fig.colorbar(plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_z)
    
    cbar_x.set_label('Arousal (X)')
    cbar_y.set_label('Dominance (Y)')
    cbar_z.set_label('Valence (Z)')
    
    scatter = ax.scatter(total_arousal, total_dominacne, total_valence, s=50, c=colors, alpha=0.6)
    
    cube(ax, [0,0,criterion_big], [criterion_small,8,8], 'g')
    cube(ax, [criterion_big,0,criterion_big], [8,8,8], 'r')
    cube(ax, [0,0,0], [criterion_small,8,criterion_small], 'b')
    cube(ax, [criterion_big,0,0], [8,8,criterion_small], 'm')
    
    ax.view_init(elev=30, azim=45)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    
    ax.set_title(f"DEAP dataset label distribution ({criterion_small}, {criterion_big})\nRED: Happy, GREEN: Calm, MAGENTA: Anger, BLUE: Sadness")
    ax.set_xlabel("Arousal (X)")
    ax.set_ylabel("Dominance (Y)")
    ax.set_zlabel("Valence (Z)")
    
    plt.show()


def rf(features, label):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print("accuracy: ", accuracy)
    
    return y_pred, X_test, y_test, accuracy

def SHAP(features, label):
    
    plt.rcParams['font.size'] = 20

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    
    # print(f"\n\nX_train\n{X_train}\n\n")
    # print(f"\n\ny_train\n{y_train}\n\n")
    
    # print(f"\n\nX_test\n{X_test}\n\n")
    # print(f"\n\ny_test\n{y_test}\n\n")
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=features.columns)

    plt.show()

def test(small, big):
    ppg, label, sampling_rate = ppg_loader4deap(f"{small}_{big}.csv")
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
    label = label.drop(nan_rows)
    # print(f"\n\nfeatures.shape: {features.shape}, label.shape: {label.shape}\n\n")
    
    # print(f"label:\n{label}")
    
    SHAP(features, label)
    
    features = features.loc[:, ["HRV_SDSD", "HRV_RMSSD", "HRV_SDNN", "HRV_MeanNN"]]
    
    y_pred, X_test, y_test, accuracy = rf(features, label)
    
    cm = confusion_matrix(y_test, y_pred)
    
    emotions = ["happy", "calm", "anger", "sadness"]
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'DEAP dataset Confusion Matrix Heatmap ({small}, {big}) Accuracy: {accuracy}%')
    plt.show()
    
    return y_pred, y_test, accuracy
    

if __name__ == "__main__":
    
    clear_emotion(0, 9)