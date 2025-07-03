"""The dataloader for DEAP dataset.

This dataset is made by BCML lab of computer engineering department of kwangwoon university
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import pickle
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class DEAPLoader(BaseLoader):
    """The data loader for the DEAP dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an DEAP dataloader.
            Args:
                data_path(str): path of a folder which stores raw frames and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |-- videos
                     |    |-- s01_trial01.avi
                     |    |-- s01_trial02.avi
                     |    |...
                     |    |-- s0n.avi
                     |-- labels
                     |    |-- s01.dat
                     |    |-- s02.dat
                     |    |...
                     |    |-- s0n.dat
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """
        Returns data directories under the path (For DEAP dataset).
        Args:
            data_path(str): e.g. "C:/path/to/RawData"
        """
        # 영상 파일은 videos 폴더 아래에 있습니다.
        video_dir = os.path.join(data_path, "videos")
        file_list = os.listdir(video_dir)
        
        if not file_list:
            raise ValueError(f"{self.dataset_name} data path empty! <{video_dir}>")
        
        dirs = list()
        for file_name in file_list:
            if not file_name.endswith('.avi'):
                continue
            # 파일명 예: s01_trial01.avi
            # subject: 파일명에서 's' 다음의 숫자 (예, 01 → 1)
            # trial: 파일명에서 'trial' 뒤의 숫자 (예, 01)
            subject = int(file_name.split('_')[0][1:])
            trial_str = file_name.split('_')[1]
            trial = int(re.sub(r'\C', '', trial_str))
            # index는 subject와 trial 정보를 이용하여 유일한 값으로 생성 (예: subject * 100 + trial)
            index = subject * 100 + trial
            video_path = os.path.join(video_dir, file_name)
            dirs.append({"index": index, "path": video_path, "subject": subject, "trial": trial, "original_name": file_name})
        
        return dirs


    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data dirs, split with begin and end values.
        """
        if begin == 0 and end == 1:  # 전체 데이터 반환
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = [data_dirs[i] for i in choose_range]
        return data_dirs_new


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        Invoked by preprocess_dataset for multi_process.
        """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        original_file_name = data_dirs[i]["original_name"]


        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(data_dirs[i]['path'])
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels (PPG 신호)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            ppg = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            # trial 번호는 파일명에는 1부터 시작하므로 내부적으로 0-based index 사용
            ppg = self.read_ppg(data_dirs[i]['path'], data_dirs[i]['subject'], data_dirs[i]['trial'] - 1)
            
        # 전처리 함수에 영상(frames)와 PPG 신호(ppg)를 전달하여 클립 생성
        frames_clips, ppg_clips = self.preprocess(frames, ppg, config_preprocess)
        
        # 전처리된 클립들을 저장하는 과정 (저장 후 생성된 파일 이름 목록 반환)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, ppg_clips, saved_filename, original_file_name)
        file_list_dict[i] = input_name_list

        # 사용한 변수들 삭제하여 메모리 해제
        del frames, ppg, frames_clips, ppg_clips, input_name_list, label_name_list


    @staticmethod
    def read_video(video_path):
        """
        Reads a video file, returns frames (T, H, W, 3).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (72, 72))  # 사이즈는 설정값에 따라 조정 (config.py에 있음)
            frames.append(frame)
        cap.release()
        return np.asarray(frames)

    @staticmethod
    def read_ppg(video_path, subject, trial_idx=0):
        """
        Reads the PPG signal from the corresponding label file.
        Args:
            video_path(str): video 파일의 전체 경로 (상위 디렉토리를 통해 labels 폴더에 접근)
            subject(int): 영상의 subject 번호 (예: 1)
            trial_idx(int): 해당 subject 내에서의 trial 인덱스 (0부터 시작)
        """
        # labels 폴더는 videos 폴더의 상위 폴더 내에 위치합니다.
        base_dir = os.path.dirname(os.path.dirname(video_path))
        label_file = os.path.join(base_dir, "labels", f"s{subject:02d}.dat")
        with open(label_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        # 예시: trial_idx번째 trial에서 39번 채널이 PPG 신호라고 가정
        ppg_data = data['data'][trial_idx, 39, :]
        return ppg_data

    
    
    