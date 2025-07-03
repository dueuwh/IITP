# core libraries
import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

# torch libraries
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# video libraries
import cv2

def split_ecg():
    base_path = "D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/filtered_ecg_signals.csv"
    file = pd.read_csv(base_path, index_col=0)
    for index in file.index.unique().tolist():
        select_index = np.array(deepcopy(file.loc[index]))
        select_index = select_index[:, 1]
        np.save(f"D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/ecg_{index}.npy", select_index)


class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for handling multi-modal data: video, time-series, and text.

    Args:
        video_data (list): A list of video tensors (e.g., each tensor shape: [frames, height, width, channels]).
        time_series_data (list): A list of time-series tensors (e.g., each tensor shape: [sequence_length, features]).
        text_data (list): A list of text strings.
        labels (list): A list of labels corresponding to the data.
        text_tokenizer (callable): A function or callable to tokenize text into tensors.
        
    Data folder structure should be:
        root
            video
                160_1.mp4
                160_2.mp4
                ...
                16n_n.mp4
                
            eeg
                160_1.mat
                160_2.mat
                ...
                16n_n.mat
                
            polar (ecg)
                160_1.csv
                160_2.csv
                ...
                16n_n.csv
                
            rppg (if exist)
                160_1.npy
                160_2.npy
                ...
                16n_n.npy
    
    """
    
    def __init__(self, video_data, time_series_data, text_data, labels, text_tokenizer):
        self.video_data = video_data
        self.time_series_data = time_series_data
        self.text_data = text_data
        self.labels = labels
        self.text_tokenizer = text_tokenizer

        assert len(video_data) == len(time_series_data) == len(text_data) == len(labels), \
            "All input data must have the same length."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get video, time-series, and text data
        video = self.video_data[idx]  # Shape: [frames, height, width, channels]
        time_series = self.time_series_data[idx]  # Shape: [sequence_length, features]
        text = self.text_data[idx]  # Raw text string
        label = self.labels[idx]  # Label

        # Tokenize the text
        text_tensor = self.text_tokenizer(text)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {
            'video': torch.tensor(video, dtype=torch.float32),
            'time_series': torch.tensor(time_series, dtype=torch.float32),
            'text': text_tensor,
            'label': label_tensor
        }
    
    def __match_length(self):
        self.video_data


if __name__ == "__main__":
    split_ecg()