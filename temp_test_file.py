# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:37:04 2024

@author: ys
"""

from scipy import io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

base_dir = "D:/home/BCML/IITP/data/What_is_this/240430_EEG Pilot test/"
ppg_dir = base_dir + "ppg/"
ppg_mat_list = os.listdir(ppg_dir)
emotion_list = []

for name in ppg_mat_list:
    if "facial" in name:
        emotion_list.append(name)
for name in emotion_list:
    mat_file = io.loadmat(ppg_dir+name)
    ppg = mat_file['ppg_current_order']
    ppg_0 = ppg[:, 0]
    ppg_1 = ppg[:, 1]
    ppg_2 = ppg[:, 2]
    
    plt.plot(ppg_0[5000:11000], label="0")
    plt.plot(ppg_1[5000:11000], label="1")
    plt.plot(ppg_2[5000:11000], label="2")
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d

# 예제 데이터 생성
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(500)

# 힐버트 변환을 사용하여 엔벨로프 계산
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)

# 피크 검출 (진폭이 0.3 이상인 피크만 검출)
peaks, _ = find_peaks(amplitude_envelope, height=0.3)

# 피크를 이용한 보간 함수 생성
interp_func = interp1d(t[peaks], amplitude_envelope[peaks], kind='cubic', fill_value="extrapolate")

# 보간을 사용하여 부드러운 엔벨로프 생성
smooth_envelope = interp_func(t)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='원본 신호')
plt.plot(t, amplitude_envelope, label='힐버트 엔벨로프', linestyle='--')
plt.plot(t, smooth_envelope, label='부드러운 엔벨로프', linestyle='-.')
plt.xlabel('시간 (초)')
plt.ylabel('진폭')
plt.title('soft envelope')
plt.legend()
plt.show()




# 0 = ecg
# 1 = ppg
# 2 = button 