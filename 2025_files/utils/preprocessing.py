import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy.signal as signal
import matplotlib.pyplot as plt

def bandpassfilter(signal):
    return signal