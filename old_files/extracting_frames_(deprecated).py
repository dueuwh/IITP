# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:47:43 2024

@author: ys
"""
import os
import cv2
import math

video_dir = './data/videos/'
video_list = os.listdir(video_dir)

frame_dir = './data/dataset_rearrange4DDAMFN/'

for name in video_list[:3]:
    video_counter = 1
    
    if name.split('.')[0] not in os.listdir(frame_dir):
        os.mkdir(frame_dir+name.split('.')[0])
    save_dir = frame_dir+name.split('.')[0]+'/'
    
    cap = cv2.VideoCapture(video_dir+name)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    exponent_10 = len(str(int(total_frames)))-1
    frame_initial_code = '10'
    for i in range(exponent_10):
        frame_initial_code += '0'
    frame_counter = int(frame_initial_code)
    print(f"{name} start, {total_frames} length")
    
    frame_counter4progress = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break
        
        cv2.imwrite(f"{save_dir}{name.split('.')[0]}_{frame_counter}.jpg", frame)
        frame_counter += 1
        frame_counter4progress += 1
        if frame_counter % 900 == 0:
            print(f"video: {video_counter}/3, frame: {round((frame_counter4progress/total_frames)*100, 1)}%")
    
    video_counter += 1
    cap.release()