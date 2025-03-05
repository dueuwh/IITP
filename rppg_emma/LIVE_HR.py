import cv2
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import matplotlib.pyplot as plt
import time
import numpy as np
from SP_test_LIVE_mp import SignalProcessing, BPM, cpu_OMIT
from scipy.signal import butter, filtfilt
from scipy.stats import trim_mean

def signal_filtering(signal, low_band=0.5, high_band=4.0, fs=30, N=2):
    #미국 기준 HR 30~240 구간 측정 가능해야함. fs : 카메라 FPS
    [b_pulse, a_pulse] = butter(N, [low_band / fs * 2, high_band / fs * 2], btype='bandpass')
    rst_signal = filtfilt(b_pulse, a_pulse, np.double(signal))

    return rst_signal


def proc(frame_queue, share_HR, save_dir):
    SNR_threshold = 0.045
    frame_count = 0
    hr_count = 0
    signal_length = 300 #한번에 사용할 frame 수.
    hr_window_size = 40 #측정에 사용할 HR 수. 
    pred_hrs = np.zeros(hr_window_size)
    fs=30 #FPS
    signal_processor = SignalProcessing(frame_queue) #class 하위 함수들에서 사용할 frame이 저장된 queue 전달

    rgb_signal = []
    
    for mean_RGB in signal_processor.extract_holistic(face_detection_interval=90): #extract_holistic에서 yield로 프레임 한장 계산해서 계속 return해줌.
        rgb_signal.append(mean_RGB)

        if len(rgb_signal) == signal_length: #rgb_signal이 가득찼을때
            pred_signal = np.squeeze(np.array(rgb_signal)).T #(3,signal_length)
            
            
            pred_signal = cpu_OMIT(pred_signal)
            pred_signal = signal_filtering(signal=pred_signal, low_band=0.5, high_band=4.0, N=2)
            pred_welch_hr, SNR, pSNR,Pfreqs, Power = BPM(data=pred_signal, fps=fs, startTime=0, minHz=0.5, maxHz=4.0, verb=False).BVP_to_BPM()

            if SNR > SNR_threshold:
                pred_hrs[hr_count] = pred_welch_hr
                hr_count+=1

            rgb_signal.pop(0) #제일 예전값 하나 제거, 새로운값 받기위함.
        
            frame_count+=1
            
            if hr_count == hr_window_size:
                print("queue left :",frame_queue.qsize())
                #이상치가 30.1이 나오는경우가 많아서 해당 부분 제거
                try:
                    share_HR.value = int(trim_mean(pred_hrs[pred_hrs>30.9],0.2)) #20% 절사평균
                except:
                    pass
                hr_count = 0
                with open(save_dir, 'w') as file:
                    file.write(str(int(trim_mean(pred_hrs[pred_hrs>30.9],0.2)))+'\n')






###
if __name__ == '__main__':
    save_dir = "D:/home/BCML/IITP/data/results/multimodal_pilottest_1.txt"
    mp.set_start_method('spawn', force=True)
    q = Queue()

    share_value = Value('i',0)
    p2 = Process(target=proc, args=(q,share_value, save_dir))
    p2.start()

    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#테스트중 카메라 연결 안될시 0->1로 변경

    cap = cv2.VideoCapture("D:/home/BCML/IITP/data/videos/multimodal_pilottest_1.mp4")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0) #어두운 환경에선 EXPOSURE, WB 자동조절 시간때문에 느리다고함. 하드웨어단에서 진행되는건지 안꺼짐.
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)##테스트중 느릴시 주석처리
    cap.set(cv2.CAP_PROP_AUTO_WB,0)##테스트중 느릴시 주석처리, DSHOW는 안되는데 v4l2는 된다는 이야기가 있음.
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(320,240), interpolation=cv2.INTER_AREA) #resolution 조정안될시 resize로 해야함. 성능은 비슷한듯.
        try:
            q.put(frame)
            frame_rst = frame.copy() #frame에 글자를 바로 써버리면, 주소를 참조하는 식이라 Queue에 넣은 이미지에 글자가 들어가버림.
            cv2.putText(frame_rst, "HR:"+str(share_value.value), (100,200), 0, 2, (0,0,255), 2) 
            cv2.imshow('frame',frame_rst)
            
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        except:
            break

    cv2.destroyAllWindows()
    cap.release()

    
