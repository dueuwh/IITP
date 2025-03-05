import os
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
  # happy / neutral / anxiety / embarrassment / hurt / sadness / anger
  # idx 0: ppg, idx 1: switch
  
  # for label
  # HJR emotion start at 2th switch (from 0 to 2)
  
  
  # for videos
  # CTB_0922 09.10 start, 35.56 end
  # HJR_0301 09.02 start, 32.33 end
  # KKJ_0829 11.57 start, 40.35 end
  # SIH_0203 08.19 start, 29.15 end
  # KNM_0408 02.42 start, 28.52 end
  # KPJ_0818 02.20 start, 32.05 end
  # PJS_0211 01.50 start, 26.22 end
  # UJS_0323 00.49 start, 28.22 end
  # CJS_0305 01.12 satrt, 27.22 end
  # JTW_0805 00.33 start, 25.44 end
  # KJH_0408 00.34 start, 25.35 end
  # SPR_0317 01.00 start, 29.10 end
  # SYG_1030 00.48 start, 26.45 end
  # EJE_0609 00.23 start, 25.13 end
  # JHS_0301 00.08 start, 26.56 end
  # OYY_0620 00.30 start, 26.26 end

# video emotion part
time_indices = {}
time_indices['CTB_0922'] = [9, 10, 35, 56]
time_indices['HJR_0301'] = [9, 2, 32, 33]
time_indices['KKJ_0829'] = [11, 57, 40, 35]
time_indices['SIH_0203'] = [8, 19, 29, 15]
time_indices['KNM_0408'] = [2, 42, 28, 52]
time_indices['KPJ_0818'] = [2, 20, 32, 5]
time_indices['PJS_0211'] = [1, 50, 26, 22]
time_indices['UJS_0323'] = [0, 49, 28, 22]
time_indices['CJS_0305'] = [1, 12, 27, 22]
time_indices['JTW_0805'] = [0, 33, 25, 44]
time_indices['KJH_0408'] = [0, 34, 25, 35]
time_indices['SPR_0317'] = [1, 0, 29, 10]
time_indices['SYG_1030'] = [0, 48, 26, 45]
time_indices['EJE_0609'] = [0, 23, 25, 13]
time_indices['JHS_0301'] = [0, 8, 26, 56]
time_indices['OYY_0620'] = [0, 30, 26, 26]


emotion_folders = ["happy", "neutral", "anxiety", "embarrassment", "hurt", "sadness", "anger"]
d231221 = [[2,4], [4,6], [6,8], [8,10], [10,12], [12,14], [14,16]]
others = [[0,2], [2,4], [4,6], [6,8], [8,10], [10,12], [12,14]]


base_dir = "C:/Users/U/Desktop/BCML/IITP/IITP_emotions/data/senior"
videos_dir = base_dir + "/videos/"
labels_dir = base_dir + "/labels/"
videos_date_list = os.listdir(videos_dir)
labels_date_list = os.listdir(labels_dir)

synchro_dir = "C:/Users/U/Desktop/BCML/IITP/IITP_emotions/data/senior/sychro/"
save_frames_dir = synchro_dir + "frames/"
save_labels_dir = synchro_dir + "labels/"

for i, date in enumerate(videos_date_list):
    video_list = os.listdir(videos_dir+date)
    label_list = os.listdir(labels_dir+date)
    
    common_name_list = [name.split('.')[0] for name in video_list]
    for name in common_name_list:
        
        frames_save_dir = save_frames_dir + date + '/' + name + '/'
        labels_save_dir = save_labels_dir + date + '/' + name + '/'
        
        if not os.path.exists(frames_save_dir):
            os.makedirs(frames_save_dir)
        
        if not os.path.exists(labels_save_dir):
            os.makedirs(labels_save_dir)
        
        if not os.path.exists(frames_save_dir+'anger'):
            for folder in emotion_folders:
                os.makedirs(frames_save_dir+folder)
        
        # print("video dir: ", videos_dir+date+name+".MP4")
        cap = cv2.VideoCapture(videos_dir+date+'/'+name+".MP4")
        
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        idx_0 = []
        idx_1 = []
        try:
            with open(labels_dir+date+'/'+name+".txt", 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    idx_0.append(float(line.split()[0]))
                    idx_1.append(float(line.split()[1]))
            label_length = len(idx_0)
            
            switch_indices = [i for i, switch in enumerate(idx_1) if switch>4.7]
            
            refined_switch_indices = []
            last_idx = 0
            for j, idx in enumerate(switch_indices):
                if j == 0:
                    last_idx = idx
                    refined_switch_indices.append(idx)
                else:
                    if idx-last_idx >= 150:
                        refined_switch_indices.append(idx)
                        last_idx = idx
                    else:                        
                        last_idx = idx
            
            
            plt.title(f"date: {date},  file name: {name},  video/label: {frame_length}/{label_length}")
            plt.plot(idx_0, label='idx 0')
            plt.plot(idx_1, label='idx 1')
            plt.scatter(refined_switch_indices, [4 for _ in range(len(refined_switch_indices))], color='g', s=30)
            plt.legend()
            plt.show()
            
            if i == 0:
                switch_idx_list = d231221
            
            else:
                switch_idx_list = others
            
            
            # total synchronization
            # label part
            total_label = idx_0[refined_switch_indices[switch_idx_list[0][0]]:refined_switch_indices[switch_idx_list[-1][1]]]
            
            # frame part
            trange = time_indices[name]
            expected_frame_length = trange[2]*60*15+trange[3]*15 - trange[0]*60*15-trange[1]*15
            gap = frame_length - expected_frame_length
            flratio = round(len(total_label)/expected_frame_length, 2)
            
            exfl4emotion = []
            for i, erange in enumerate(switch_idx_list):
                exfl4emotion.append(int((refined_switch_indices[erange[1]]-refined_switch_indices[erange[0]])/flratio))
            
            gap_frame = expected_frame_length - sum(exfl4emotion)
            if gap_frame < 0:
                for i in range(abs(gap_frame)):
                    exfl4emotion[i%7] -= 1
            elif gap_frame == 0:
                pass
            else:
                for i in range(gap_frame):
                    exfl4emotion[i%7] += 1
        
            print(f"========================================\n\nname: {name}")
            print("corresponding label length: ", len(total_label))
            print("expected frame length: ", expected_frame_length)
            print("label/ex frame: ", flratio)
            print("frame_length: ", frame_length)
            print("gap: ", gap)
            print("exfl4emotion: ", exfl4emotion)
            print("exfl4emotion sum: ", sum(exfl4emotion))
            print("emotion sum gap: ", gap_frame)
            print("expected from label == expected frome video: ", sum(exfl4emotion)==expected_frame_length)
            print("\n\n")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, trange[0]*60*15+trange[1]*15)
            frame_counter = 0
            emotion_range_counter = 0
            emotion_range_counter_idx = 0
            
            # emotion part
            for i, erange in enumerate(switch_idx_list):
                # label part
                save_np = idx_0[refined_switch_indices[erange[0]]:refined_switch_indices[erange[1]]]
                save_np = np.array(save_np)
                np.save(save_labels_dir+date+'/'+name+'/'+emotion_folders[i]+'.npy', save_np)
                
                # frame part
                if i == 0:
                    emotion_range_counter = exfl4emotion[emotion_range_counter_idx]
                    emotion_range_counter_idx += 1
                else:
                    emotion_range_counter += exfl4emotion[emotion_range_counter_idx]
                    emotion_range_counter_idx += 1
                
                while True:
                    if frame_counter == emotion_range_counter:
                        break
                    
                    ret, frame = cap.read()
                    cv2.imwrite(save_frames_dir+date+'/'+name+'/'+emotion_folders[i]+'/'+str(1000000+frame_counter%emotion_range_counter)+".jpg" , frame)
                    frame_counter += 1
                    if frame_counter % 900 == 0:
                        print(f"{round(frame_counter/expected_frame_length*100, 2)}%")
                    
            
        except FileNotFoundError:
            print(f"file not found: {date}/{name}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    