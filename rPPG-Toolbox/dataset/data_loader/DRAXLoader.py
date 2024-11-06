"""The dataloader for DRAX dataset.

This dataset is made by BCML lab of computer engineering department of kwangwoon university
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class DRAXLoader(BaseLoader):
    """The data loader for the DRAX dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an DRAX dataloader.
            Args:
                data_path(str): path of a folder which stores raw frames and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                * some of videos have 'mov' format
                -----------------
                     RawData/
                     |-- videos
                     |    |-- Subject1_file1.avi
                     |    |-- Subject1_file2.avi
                     |    |...
                     |    |-- SubjectN_fileN.mov
                     |-- labels
                     |    |-- Subject1_file1.csv
                     |    |-- Subject1_file2.csv
                     |    |...
                     |    |-- SubjectN_fileN.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For DRAX dataset).
        Args:
            data_path(str): C:/Users/U/Desktop/BCML/Drax/PAPER/data
        """
        file_list = os.listdir(data_path + os.sep + "videos/")
        
        subject_names = [name.split('_')[0] for name in file_list]
        subject_names = list(set(subject_names))
        
        subject_num = {}
        for i, name in enumerate(subject_names):
            subject_num[name] = i
        
        if not file_list:
            raise ValueError(f"{self.dataset_name} data path empty! <{data_path}>")
            
        dirs = list()
        for file_name in file_list:
            append_path = data_path + os.sep + f"videos/{file_name}"
            subject_name = file_name.split('_')[0]
            subject_file_num = file_name.split('_')[1]
            subject_trail_val = str(subject_num[subject_name]+1)+subject_file_num
            index = int(subject_trail_val)
            subject = int(subject_num[subject_name]+1)
            dirs.append({"index":index, "path":append_path, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(data_dirs[i]['path'])
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(data_dirs[i]['path'])
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        # input_name_list, label_name_list = self.save(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list
        
        del frames, bvps, frames_clips, bvps_clips, input_name_list, label_name_list

    @staticmethod
    def read_video(frame_dir):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        img_list = sorted(os.listdir(frame_dir))
        for img in img_list:
            frame = cv2.imread(frame_dir+'/'+img)
            if frame.shape[0] == 720:
                frame = frame[60:660, 80:680, :]
            else:
                frame = frame[:, 140:-140, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9, interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
        return np.asarray(frames)

    @staticmethod
    def read_wave(frame_dir):
        """Reads a bvp signal file."""
        ppg_dir = frame_dir.replace("frames", "labels")+'.npy'
        bvp = np.load(ppg_dir)
        return bvp

    
    
    