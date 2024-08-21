import numpy as np
import math
import csv
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import plotly.graph_objects as go
import heartpy as hp
from scipy.signal import stft, hilbert, find_peaks
from scipy import signal, io
import copy
from PyEMD import EMD
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import neurokit2 as nk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math as m
from llm_tuning import LLM

import pickle


full_emotion_list = ["focus", "happy", "neutral", "anxiety", "confusion", 
                "pain", "depression", "anger"]


class simple_kalman_filter():
    def __init__(self, q, a, p):
        self.q = q
        self.a = a
        self.p = p


def bpf(data, sampling_frequency, low_freq=0.75, high_freq=3.0, order=3):
    nyquist_freq = 0.5 * sampling_frequency
    low_freq_normalized = low_freq/nyquist_freq
    high_freq_normalized = high_freq/nyquist_freq
    b, a = signal.butter(order, [low_freq_normalized, high_freq_normalized], 
                         fs=sampling_frequency, btype='band')
    filtered_x = signal.lfilter(b, a, data)
    return filtered_x


# ========================= intrasubject version code =========================
# col 1 = button, col 2 = ppg
def ppg_loader4legacy(base_dir, ploting=False, data_point=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]], 
               ploting_go=False, sampling_frequency=100):
    raw_list = os.listdir(base_dir)
    
    # select intrasubject
    labellist = [name for name in raw_list if '.txt' in name]
    label_list = [name for name in labellist if "KSK" not in name and "KKJ" in name]
    
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
        if ploting:
            plt.plot(line_0, label="col 0")
            plt.plot(line_1, label="col 1")
            plt.legend()
            plt.title(f"{label} plot")
            plt.show()
            
        label_dic[label][0] = np.array(line_0)
        label_dic[label][1] = np.array(line_1)
    
    data_split_points_raw = np.where(label_dic[label][1]>=5)[0]
    
    point_num = data_point[-1][1]+1
    
    data_split_points = [[] for _ in range(point_num)]
    old = data_split_points_raw[0]
    split_point = 0
    data_split_points[split_point].append(old)
    
    for num in data_split_points_raw:
        if num - old > 1000:    
            data_split_points[split_point].append(old)
            split_point += 1
            if split_point < point_num:
                old = num
                data_split_points[split_point].append(old)
        else:
            old = num
    ppg_idx = {}
    data_point_idx = 0
    for emotion in emotion_list:
        data_range = data_point[data_point_idx]
        start = data_split_points[data_range[0]][1]
        end = data_split_points[data_range[1]][0]
        ppg_idx[emotion] = [start, end]
        data_point_idx += 1
    
    if ploting:
        plt.plot(label_dic[label][0], label="before the bpf")
        plt.title("ppg bpf before")
        plt.legend()
        plt.show()
    
    label_dic[label][0] = bpf(label_dic[label][0], sampling_frequency=sampling_frequency)
    
    if ploting:
        plt.plot(label_dic[label][0], label="after the bpf")
        plt.title("ppg bpf after")
        plt.legend()
        plt.show()
    
    
    output = {}
    for emotion in emotion_list:
        output[emotion] = label_dic[label][0][ppg_idx[emotion][0]:ppg_idx[emotion][1]]
    
    return output


def ppg_loader4pilottest(plot_verbose=0):
    # video 1: neutral, depression, anger, happy
    # video 2: neutral, depression, anger, happy
    # sampling rate: 600
        
    pilot_emotion = ['neutral', 'depression', 'anger', 'happy']

    base_dir = "D:/home/BCML/IITP/data/What_is_this/240430_EEG Pilot test/"
    ppg_dir = base_dir + "ppg/"
    ppg_mat_list = os.listdir(ppg_dir)
    emotion_list = []

    for index, name in enumerate(ppg_mat_list):
        if "facial" in name:
            emotion_list.append(name)
            
    output = {}
            
    for index, name in enumerate(emotion_list):
        mat_file = io.loadmat(ppg_dir+name)
        ppg = mat_file['ppg_current_order']
        
        if plot_verbose:
            ppg_0 = ppg[:, 0]
            ppg_1 = ppg[:, 1]
            ppg_2 = ppg[:, 2]
        
            plt.plot(ppg_0[5000:11000], label="0")
            plt.plot(ppg_1[5000:11000], label="1")
            plt.plot(ppg_2[5000:11000], label="2")
            plt.legend()
            plt.show()
    
        output[pilot_emotion[index]] = copy.deepcopy(ppg[:, 1])
    
    return output, 600
        

def split_train_test(data, ratio=0.2, loc_place=False):
    
    """
    if loc_place is False, test dataset will be sampled from end of the data
    else, test dataset will be sampled from start of the data
    """
    output_train = {}
    output_test = {}
    for emotion in emotion_list:
        temp = data[emotion]
        split_temp = int(len(temp)*ratio)
        if loc_place:
            output_train[emotion] = data[emotion][split_temp:]
            output_test[emotion] = data[emotion][:split_temp]
        else:
            output_train[emotion] = data[emotion][:-split_temp]
            output_test[emotion] = data[emotion][-split_temp:]
    
    return output_train, output_test


class feature_extraction:
    def __init__(self, data, feature_list, data_sampling_rate, data_emotion_list):
        self.data = data
        self.sampling_rate = data_sampling_rate
        self.feature_list = feature_list
        self.one_step = 1/self.sampling_rate
        self.x_axis = []
        self.emotion_list = data_emotion_list
        
        data_length = len(self.data)
        index = 0
        for _ in range(data_length):
            self.x_axis.append(index)
            index += self.one_step # ms
    
    """
        Feature references: 
            https://dl.acm.org/doi/pdf/10.1145/3009960.3009962
        
        All the feature extraction functions should be modified to process real
        -time input data. Currently, these functions only deal with total
        length of input dataset.
        
        -> One frame of ppg / rppg is input of 'get' funtion.
        -> Each functions calculate the feature of the frame
        -> 'get' function gathers features as dictionary and return it as
            the result
    """
    
    def rr(self, plot_verbose=1, peaks_distance=10, envelope_window_radius=10,
           envelope_toggle = 0, envelope_smoothing='cubic',
           envelope_interp_fill="extrapolate", rulebased_toggle = 0,
           gradient_toggle=0, gradient_window=10, convexhull_toggle=0,
           convexhull_window=5, neurokit_toggle=1, frequency_toggle=0,
           bpf=(0.5, 4)):
        
        ### Except the neurokit2 extraction, the others are deprecated.
        ### I leave it for further study and some ensemble model with neurokit2
        
        """
            Additional functions?
                1. Interval average post-processing
                    Keep calculating RR interval average and finding anomalous
                    empty space
                2. Interpolation function for all methods using copped window
        """
        
        """
            plot_verbose: Drawing the rr plot or not (1/0 or True/False)
            
            envelope_window_radius: window radius for finding the largest
                                    amplitude of the envelope
            
            envelope_toggle: Enable or disable envelope method / 0 or 1
            
            envelope_smoothing: The type of envelope detection result
            
            envelope_interp_fill: The method of filling interpolated envelope
                                  value.
            
            * envelope method is not working well. It's better to use gradient-
              based method. I leave it for futher research.
              
             rulebased_toggle: Enable or disable rule-based method / 0 or 1
             
            * Rule-based method: checking current peak and next 2 peaks.
              This method checks the next 4 condition:
                  1. Is the gradient between first one and second one is minus
                     && the gradient between second one and third one is plus
                  2. Is the gradient 
                  3. Is the gradient between first one and third one is
                     smoother than the others
                  3. Is the condition 1 is not satisfied and 
              
              -- Implmenting gradient-based method is in progress
             
            gradient_toggle: Enable or disable gradient-based method / 0 or 1
            
            gradient_window: The window for gradient method utilized for crop
                             peaks
            
            convexhull_toggle: Enable or disable convexhull method / 0 or 1
            
            convexhull_window: window for convexhull utilized for crop peaks
            
            * convexhull method: convexhull method implemented in scipy
            
            neurokit_toggle: Utilizing neurokit2 library or not / 0 or 1
            
            frequency_toggle: Enable or disalbe frequency-based method / 0 or 1
            
            bpf: band-pass filter frequency tuple (low, high)
        """
        
        self.peaks_index = {}
        
        for key in self.data.keys():
            peaks, _ = find_peaks(self.data[key], height=0.1, distance=peaks_distance)
            peaks_inv, _ = find_peaks(-self.data[key], height=0.1, distance=peaks_distance)
            
            x = self.x_axis[:len(self.data[key])]
            peaks_raw = [self.data[key][idx] for idx in peaks]
            peaks_length = len(peaks)
            
            if envelope_toggle:
                analytic_signal = hilbert(self.data[key])
                amplitude_envelope = np.abs(analytic_signal)
                
                # smoothed envelope
                # envelope_peaks, _ = find_peaks(amplitude_envelope, height=0.4, distance=envelope_window_radius)
                
                # interp_func = interp1d([x[idx] for idx in envelope_peaks], [self.data[key][idx] for idx in envelope_peaks],
                #                        kind=envelope_smoothing, fill_value=envelope_interp_fill)
                
                # smoothed_envelope = interp_func(x)
            
            if rulebased_toggle:
                peaks_rulebased = []
                
                peaks_length = len(peaks)
                idx = 0
                
                while True:
                    if idx == peaks_length-1:
                        break
                    try:
                        current = self.data[key][peaks[idx]]
                        current_1 = self.data[key][peaks[idx+1]]
                        current_2 = self.data[key][peaks[idx+2]]
                        
                        c_1_1 = current_1 - current
                        c_1_2 = current_2 - current_1
                        g_2_1 = c_1_1 / (peaks[idx+1]-peaks[idx])
                        g_2_2 = c_1_2 / (peaks[idx+2]-peaks[idx+1])
                        g_3_1 = (current_2 - current) / (peaks[idx+2] - peaks[idx])
                        condition_1 = c_1_1 < 0 and c_1_2 > 0
                        condition_2 = g_2_1 > 1/600 and abs(g_2_2) > 1/600
                        condition_3 = g_2_1 > abs(g_3_1) or g_2_2 > abs(g_3_1)
                        
                    except IndexError:
                        current_2 = self.data[key][peaks[idx]]
                        current_1 = self.data[key][peaks[idx-1]]
                        current = self.data[key][peaks[idx]-2]
                        
                        c_1_1 = current_1 - current
                        c_1_2 = current_2 - current_1
                        g_2_1 = c_1_1 / (peaks[idx-1]-peaks[idx-2])
                        g_2_2 = c_1_2 / (peaks[idx]-peaks[idx-1])
                        g_3_1 = (current_2 - current) / (peaks[idx] - peaks[idx-2])
                        condition_1 = c_1_1 < 0 and c_1_2 > 0
                        condition_2 = g_2_1 > 1/600 and abs(g_2_2) > 1/600
                        condition_3 = g_2_1 > abs(g_3_1) and g_2_2 > abs(g_3_1)
                    
                    if not (condition_1 and condition_3) and not condition_2:
                        peaks_rulebased.append(peaks[idx])
                    
                    idx += 1
                
            if gradient_toggle:
                
                peaks_gradient = []
                
                peaks_raw_copy = copy.deepcopy(peaks_raw)
                peaks_copy = copy.deepcopy(peaks)
                
                iter_index = 0
                iter_count = 0
                last = False
                while True:
                    first = True
                    window_start = gradient_window * iter_count
                    window_end = gradient_window * (iter_count + 1)
                    
                    iter_count += 1
                    
                    try:
                        temp_amplitude_window = peaks_raw_copy[window_start : window_end]
                        temp_peakindex_window = peaks[window_start : window_end]
                    except IndexError:
                        temp_amplitude_window = peaks_raw_copy[window_start:]
                        temp_peakindex_window = peaks[window_start:]
                        last = True
                    
                    max_amplitude = max(temp_amplitude_window)
                    max_amplitude_index = temp_peakindex_window[temp_amplitude_window.index(max_amplitude)]
                    
                    if first:
                        peaks_gradient.append(max_amplitude_index)
                        first = False
                    else:
                        while True:
                            break
                    
                    
                    if last:
                        break
            
            if convexhull_toggle:
                
                peaks_convex = []                
                                
                iter_count = 0
                
                peaks_raw_copy = copy.deepcopy(peaks_raw)
                peaks_copy = copy.deepcopy(peaks)
                
                last = False
                while True:
                    window_start = convexhull_window * iter_count
                    window_end = convexhull_window * (iter_count + 1)
                    
                    iter_count += 1
                    
                    try:
                        temp_amplitude_window = peaks_raw_copy[window_start : window_end]
                        temp_peakindex_window = peaks[window_start : window_end]
                    except IndexError:
                        temp_amplitude_window = peaks_raw_copy[window_start:]
                        temp_peakindex_window = peaks[window_start:]
                        last = True
                        
                    point_2d = np.concatenate((np.array(temp_peakindex_window).reshape(-1, 1), np.array(temp_amplitude_window).reshape(-1, 1)), axis=1)
                    plt.scatter(point_2d[:, 0], point_2d[:, 1])
                    plt.show()
                    hull = ConvexHull(point_2d)
                    
                    if last:
                        break
            
            # only one that working well
            if neurokit_toggle:
                self.signals, info = nk.ppg_process(self.data[key], sampling_rate=self.sampling_rate)
                nk.ppg_plot(self.signals, info)
                
                peaks_neurokit2 = []
                for i in range(len(self.signals.PPG_Peaks)):
                    if self.signals.PPG_Peaks[i] == 1:
                        peaks_neurokit2.append(i)
                
                self.peaks_index[key] = peaks_neurokit2
                
            if frequency_toggle:
                # frequency-based peak detection function
                # Finding frequency has largest amlitude and corresponding time
                pass
                
        
            if plot_verbose:
                fig = plt.figure(figsize=(12,6))
                ax = fig.add_subplot(1,1,1)
                
                if neurokit_toggle:
                    ax.scatter([x[idx] for idx in peaks_neurokit2], [self.data[key][idx] for idx in peaks_neurokit2], label="Neurokit2 peaks",
                               c='green', zorder=6, s=50)
                
                if gradient_toggle:
                    pass
                
                if envelope_toggle:
                    # ploting Envelope detection
                    ax.plot(x, amplitude_envelope, label='Envelope', linestyle='dashed')
                    
                    # scattering envelope amplitude at peack points
                    ax.scatter([x[idx] for idx in peaks], [amplitude_envelope[idx] for idx in peaks], label="Envelope", s=10, c='m', zorder=4)
                    
                    # ploting Smoothed envelope detection
                    # ax.plot(x, smooth_envelope, label="Smoothed envelope", linestyle='dotted')

                # ploting original signal and peaks
                # plt.scatter([x[idx] for idx in peaks_inv], [self.data[key][idx] for idx in peaks_inv], s=15, c='g', label='valley', zorder=2)
                ax.plot(x, self.data[key], label='PPG')
                ax.scatter([x[idx] for idx in peaks], peaks_raw, s=15, c='r', label='Scipy peaks', zorder=7)
                if rulebased_toggle:
                    ax.scatter([x[idx] for idx in peaks_rulebased], [self.data[key][idx] for idx in peaks_rulebased], s=25, c='blue', label='peak_gradient')
                # if envelope_toggle:
                #     ax.scatter([x[idx] for idx in peaks_envelope], [self.data[key][idx] for idx in peaks_envelope], s=25, c='blue', label='peak_envelope', zorder=3)
                plt.xlabel('seconds')
                plt.ylabel('uV')
                plt.title(f'emotion: {key}, peak detection')
                plt.legend()
                # plt.xlim(51, 68)
                ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                plt.show()
    
    def sdnn(self, plot_verbose=1):
        self.rr_list = {}
        self.sdnn = {}
        for key in self.data.keys():
            peaks = self.peaks_index[key]
            self.rr_list[key] = []
            for i in range(len(peaks)-1):
                self.rr_list[key].append((peaks[i+1]-peaks[i])*self.one_step)
                
            self.sdnn[key] = np.std(self.rr_list[key])
            
        if plot_verbose:
            x = self.emotion_list
            sdnn = []
            for key in self.data.keys():
                sdnn.append(self.sdnn[key])
            plt.bar(x, sdnn)
            plt.title(f"sdnn")
            plt.xlabel("Emotions")
            plt.ylabel("sdnn")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.sdnn[self.emotion_list[i]],
                         f"{round(self.sdnn[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
            
        return self.sdnn
    
    def rmssd(self, plot_verbose=1):
        self.rmssd = {}
        for key in self.data.keys():
            self.rmssd[key] = np.sqrt(np.mean(np.diff(self.rr_list[key])**2))
        if plot_verbose:
            x = self.emotion_list
            rmssd = []
            for key in self.data.keys():
                rmssd.append(self.rmssd[key])
            plt.bar(x, rmssd)
            plt.title(f"RMSSD")
            plt.xlabel("Emotions")
            plt.ylabel("RMSSD")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.rmssd[self.emotion_list[i]],
                         f"{round(self.rmssd[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
        
        return self.rmssd
    
    def nn50(self, plot_verbose=1):
        self.nn50 = {}
        for key in self.data.keys():
            self.nn50[key] = np.sum(np.abs(np.diff(self.rr_list[key]) > 0.05))
        
        if plot_verbose:
            x = self.emotion_list
            nn50 = []
            for key in self.data.keys():
                nn50.append(self.nn50[key])
            plt.bar(x, nn50)
            plt.title(f"NN50")
            plt.xlabel("Emotions")
            plt.ylabel("NN50 (count)")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.nn50[self.emotion_list[i]],
                         f"{round(self.nn50[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
                           
        return self.nn50
    
    def pnn50(self, plot_verbose=1):
        self.pnn50 = {}
        for key in self.data.keys():
            self.pnn50[key] = (self.nn50[key] / len(self.rr_list[key])) * 100
        
        if plot_verbose:
            x = self.emotion_list
            pnn50 = []
            for key in self.data.keys():
                pnn50.append(self.pnn50[key])
            plt.bar(x, pnn50)
            plt.title(f"pNN50")
            plt.xlabel("Emotions")
            plt.ylabel("pNN50 (%)")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.pnn50[self.emotion_list[i]],
                         f"{round(self.pnn50[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
        
        return self.pnn50
    
    def edd1(self, plot_verbose=1):
        euclidean_1 = {}
        for key in self.data.keys():
            euclidean_1[key] = []
            
            for value in self.data[key]:
                euclidean_1[key].append(0)
        return euclidean_1
    
    def edd2(self, plot_verbose=1):
        return 0
    
    # for the window size of one period, 
    
    def __kfd(self, key, idx, plot_verbose=1):
        window=self.sampling_rate
        temp_window = self.data[key][idx*window:(idx+1)*window]
        total_edd = []
        max_edd = []
        for i in range(len(temp_window)-1):
            total_edd.append(((temp_window[i+1]-temp_window[i])**2)/self.one_step)
            max_edd.append(((temp_window[i+1]-temp_window[0])**2)/(self.one_step*(i+1)))
        
        L = window / self.sampling_rate
        a = np.mean(total_edd)
        d = max(max_edd)
        return (m.log10(L/a)/m.log10(d/a))
            
        
    def kfd(self, plot_verbose=1):
        
        for key in self.data.keys():
            for idx in range(len(self.data[key])):
                kfd = self.__kfd(key, idx)
            
        return 0
    
    def stft(self, plot_verbose=1):
        return 0
    
    def spectral_coherence(self, plot_verbose=1):
        return 0
    
    def sdann(self, plot_verbose=1):
        
        """
            if the length of each PPG/rPPG segment for each emotion is
            5 minutes or shorter, the SDANN output will probably be NaN
            Long emotion dataset is required.
        """
        
        self.sdann = {}
        for key in self.data.keys():
            segment_length = 2 * 60  # seconds
            num_segments = int(np.floor(len(self.rr_list[key]) / (segment_length * sampling_rate)))
            
            segment_means = []
            for i in range(num_segments):
                segment_start = i * segment_length * self.sampling_rate
                segment_end = (i + 1) * segment_length * self.sampling_rate
                segment_mean = np.mean(self.rr_list[key][int(segment_start):int(segment_end)])
                segment_means.append(segment_mean)
            
            self.sdann[key] = np.std(segment_means)
        
        if plot_verbose:
            x = self.emotion_list
            sdann = []
            for key in self.data.keys():
                sdann.append(self.sdann[key])
            plt.bar(x, sdann)
            plt.title(f"sdann")
            plt.xlabel("Emotions")
            plt.ylabel("sdann")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.sdann[self.emotion_list[i]],
                         f"{round(self.sdann[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
        
        return sdann
    
    def hrv_triangular_index(self, plot_verbose=1, bin_width=7.8125):
        self.hrv_triangular_index = {}
        for key in self.data.keys():
            hist, bin_edges = np.histogram(self.rr_list[key], bins=np.arange(min(self.rr_list[key]),
                                           max(self.rr_list[key]) + bin_width, bin_width))
            
            y_max = np.max(hist)
            
            d_total = np.sum(hist)
            
            self.hrv_triangular_index[key] = d_total / y_max
        
        if plot_verbose:
            x = self.emotion_list
            hrv_triangular_index = []
            for key in self.data.keys():
                hrv_triangular_index.append(self.hrv_triangular_index[key])
            print("type: ", type(hrv_triangular_index))
            plt.bar(x, hrv_triangular_index)
            plt.title(f"hrv_triangular_index")
            plt.xlabel("Emotions")
            plt.ylabel("hrv_triangular_index")
            for i in range(len(self.data.keys())):
                plt.text(x[i], self.hrv_triangular_index[self.emotion_list[i]],
                         f"{round(self.hrv_triangular_index[self.emotion_list[i]], 7)}", ha='center', va='bottom', size=10)
            plt.legend()
            plt.show()
        
        return self.hrv_triangular_index
    
    def sdmm(self, plot_verbose=1):
        return 0
    
    def sdsd(self, plot_verbose=1):
        return 0
    
    def tinn(self, plot_verbose=1):
        return 0
    
    def vlf(self, plot_verbose=1):
        return 0
    
    def lf(self, plot_verbose=1):
        return 1
    
    def ulf(self, plot_verbose=1):
        return 0
    
    def hf(self, plot_verbose=1):
        return 1
    
    def stress_index(self, plot_verbose=1):
        return 0
    
    def get(self, plot_verbose=1):
        
        # time domain features
        rr = self.rr()
        sdnn = self.sdnn()
        rmssd = self.rmssd()    
        nn50 = self.nn50()
        pnn50 = self.pnn50()
        sdann = self.sdann()
        hrv_triangular_index = self.hrv_triangular_index()
        sdmm = self.sdmm()
        sdsd = self.sdsd()
        tinn = self.tinn()
        euclidian_distance_d1 = self.edd1()
        euclidian_distance_d2 = self.edd2()
        kfd = self.kfd()
        stft = self.stft()
        
        # frequency domain feature
        vlf = self.vlf()
        lf = self.lf()
        ulf = self.ulf()
        hf = self.hf()
        s_balance = lf/hf
        
        spectral_coherence = self.spectral_coherence()
        
        features = {}
        
        for emotion in self.data.keys():
            features[emotion] = {}
            for feature_name in self.feature_list:
                features[emotion][feature_name] = locals()[feature_name]
        
        return features


def change_dict2np(input_dict):
    outputnp = "dictionary[key][index] -> (features, index) shape ndarray * key"
    return outputnp


def feature_extraction_pipeline(data_window, stride, total_dataset, sampling_rate,
                                feature_list, data_emotion_list):
    
    """
        This function conducts feature extraction repeatly over the all input data
    """
    feature_extractor = feature_extraction(data_window, feature_list, sampling_rate, data_emotion_list)
    
    total_output = {}
    for key in total_dataset.keys():
        set_data = data_window[key]
        total_output[key] = {}
        for i in range((len(set_data)-data_window)/stride + 1):
            total_output[key][i] = feature_extractor.get(set_data[i*(data_window+stride) : (i+1)*(data_window+stride)])
    
    return total_output
        


class animation_ploting:
    def __init__(self):
        return 0

class feature_analysis:
    
    """
        Candidates
        
        1. SHAP
        
        2. ELI5
        
        3. Anchor
        
        4. Individual Conditional Expectation (ICE)
        
        5. ALE
        
        6. Chi-Square Test
        
    """
    
    def __init__(self, result):
        self.result = result

def RandomForest():
    return 0

def SVM():
    return 0

def EM():
    return 0

def DNN():
    return 0

def LLM():
    llama_3 = LLM("llama_3")
    return 0

if __name__ == "__main__":
    # pilot_test label
    # col 0 = ecg
    # col 1 = ppg
    # col 2 = switch
    
    ### rPPG extraction
    
    
    
    
    ### Loading dataset
    
    pilot_test, sampling_rate = ppg_loader4pilottest()
    
    ### Feature extraction
    
    feature_list = ["rr", "sdnn", "rmssd", "nn50", "pnn50", "sdann",]
    window_length = 10  # seconds
    stride = 1  # seconds
    emotion_list = ["happy", "neutral", "depression", "anger"]
    
    total_features = feature_extraction_pipeline(window_length, stride, pilot_test, 
                                                 sampling_rate, feature_list, emotion_list)
    
    
    ### Sample model test
    
    
    ### Feature analysis
    
    
    ### LLM model test
    
    
    ### Result ploting
    
    



    
# =============================================================================
#     feature_list = ["rr", "hrv", "sdnn", "rmssd", "nn50", "pnn50",
#                     "euclidian_distance_d1", "euclidian_distance_d2",
#                     "kfd", "stft", "spectral_coherence", "sdann",
#                     "hrv_triangular_index", "sdmm", "sdsd", "tinn",
#                     "vlf", "lf", "ulf", "hf", "s_balance"]
#     sf = 30
#     stft_window = sf*10  # stft window is 10 seconds
#     print("cwd: ", os.getcwd())
#     base_dir = "./data/labels/"
#     ppg = ppg_loader(base_dir, ploting=False, sampling_frequency=sf)
#     train_ppg, test_ppg = split_train_test(ppg)
#     hrv_0emotion_train = hrv_extractor(train_ppg[emotion_list[0]])
# =============================================================================
    
    
    