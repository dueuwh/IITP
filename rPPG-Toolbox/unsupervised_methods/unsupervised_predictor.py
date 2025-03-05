"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
from collections.abc import Iterable
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
import os
import sys
from scipy.interpolate import interp1d
from copy import deepcopy

def replace_nan_and_inf_with_interpolation(array):
    """
    입력된 numpy 배열에서 각 열(column)별로 inf와 nan 값을 선형 보간으로 대체합니다.
    - 처음이나 끝에 inf 또는 nan이 있으면 가장 가까운 유효한 값을 사용합니다.
    - inf와 nan의 총 개수가 전체 데이터의 30%를 넘으면 원본 배열을 반환합니다.

    Parameters:
        array (numpy.ndarray): N x 3 형태의 2차원 numpy 배열.

    Returns:
        numpy.ndarray: inf와 nan이 선형 보간 또는 가장 가까운 값으로 대체된 배열.
    """
    # 입력 배열이 2차원인지 확인
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Input array must be a 2D array with shape (N, 3).")

    # 복사본 생성 (원본 배열을 수정하지 않기 위해)
    result = array.copy()

    # 각 열(column)에 대해 처리
    for col in range(result.shape[1]):
        column_data = result[:, col]

        # inf와 nan 위치 확인
        invalid_mask = np.isnan(column_data) | np.isinf(column_data)

        # inf와 nan의 총 개수가 전체 데이터의 30%를 넘는 경우
        if np.sum(invalid_mask) > 0.3 * len(column_data):
            print(f"Too many invalid values in column {col} (more than 30%). Returning the original array.")
            return array, True  # 원본 배열 반환

        # 유효한 값의 인덱스와 값 추출
        valid_indices = np.where(~invalid_mask)[0]
        valid_values = column_data[~invalid_mask]

        # 유효한 값이 없으면 원본 배열 반환
        if len(valid_indices) == 0:
            raise ValueError(f"Column {col} contains only NaN or Inf values, cannot interpolate.")

        # 처음이나 끝에 inf 또는 nan이 있는 경우 가장 가까운 유효한 값으로 대체
        if invalid_mask[0]:  # 처음 값이 invalid인 경우
            column_data[0] = valid_values[0]
        if invalid_mask[-1]:  # 마지막 값이 invalid인 경우
            column_data[-1] = valid_values[-1]

        # 선형 보간 함수 생성
        interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")

        # 보간을 통해 inf와 nan 값을 대체
        column_data[invalid_mask] = interp_func(np.where(invalid_mask)[0])

        # 결과를 다시 저장
        result[:, col] = column_data

    return result, False

def amplitudeSelectiveFiltering(C_rgb, amax = 0.002, delta = 0.0001):
    '''
    Input: Raw RGB signals with dimensions 3xL, where the R channel is column 0
    Output: 
    C = Filtered RGB-signals with added global mean, 
    raw = Filtered RGB signals
    '''

    L = C_rgb.shape[1]
    C = (1/(np.mean(C_rgb,1)))


    #line 1
    C = np.transpose(np.array([C,]*(L)))* C_rgb -1
    #line 2       
    F = abs(np.fft.fft(C,n=L,axis=1)/L) #L -> C_rgb.shape[0]

    #line 3   
    W = (delta / np.abs(F[0,:])) #F[0,:]  is the R-channel
    
    #line 4
    W[np.abs(F[0,:]<amax)] = 1
    W = W.reshape([1,L])

    #line 5
    Ff = np.multiply(F,(np.tile(W,[3,1])))
    
    #line 6
    C = np.transpose(np.array([(np.mean(C_rgb,1)),]*(L))) * np.abs(np.fft.ifft(Ff)+1)
    raw = np.abs(np.fft.ifft(Ff)+1)
    return C.T, raw.T

def safe_add(target_list, data):
    """
    Safely add data to a list. If data is iterable (excluding strings and bytes), use extend.
    Otherwise, use append.
    
    Args:
        target_list (list): The list to which data is added.
        data (iterable or scalar): The data to add to the list.
    """
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        target_list.extend(data)
    else:
        target_list.append(data)

def append_to_npy(file_path, new_data):
    """Appends new data to an existing .npy file or creates a new one if it doesn't exist."""
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        combined_data = np.concatenate((existing_data, new_data))
    else:
        combined_data = new_data
    np.save(file_path, combined_data)

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    
    gt_hr_temp = []
    pre_hr_temp = []
    SNR_temp = []
    macc_temp = []
    
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    previous_batch_window = []
    for i_batch, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            
            for i in range(72):
                for j in range(72):
                    temp_output = amplitudeSelectiveFiltering(data_input[:, i, j, :].squeeze().T)
                    temp_output, toomanyError = replace_nan_and_inf_with_interpolation(temp_output[1])
                    if toomanyError:
                        pass
                    else:
                        data_input[:, i, j, :] = np.array(temp_output)
            try:
                if method_name == "POS":
                    BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "CHROM":
                    BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "ICA":
                    BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "GREEN":
                    BVP = GREEN(data_input)
                elif method_name == "LGI":
                    BVP = LGI(data_input)
                elif method_name == "PBV":
                    BVP = PBV(data_input)
                elif method_name == "OMIT":
                    BVP = OMIT(data_input)
                else:
                    raise ValueError("wrong unsupervised method name!")
                previous_batch_window = deepcopy(BVP)
            except:
                BVP = previous_batch_window
                
            
            if not os.path.exists(f"{config.INFERENCE.SAVE_PATH}/{method_name}"):
                os.mkdir(f"{config.INFERENCE.SAVE_PATH}/{method_name}")

            if config.INFERENCE.SAVE_BVP:
                np.save(f"{config.INFERENCE.SAVE_PATH}/{method_name}/bvp_{i_batch}_{idx}.npy", BVP)

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size        
                
            gt_hr = []
            pre_hr = []
            SNR = []
            macc = []
            
            gt_hr_temp = []
            pre_hr_temp = []
            SNR_temp = []
            macc_temp = []
                
            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    safe_add(gt_hr_peak_all, gt_hr)
                    safe_add(predict_hr_peak_all, pre_hr)
                    safe_add(SNR_all, SNR)
                    safe_add(MACC_all, macc)
                    
                    safe_add(gt_hr_temp, gt_hr)
                    safe_add(pre_hr_temp, pre_hr)
                    safe_add(SNR_temp, SNR)
                    safe_add(macc_temp, macc)
                    
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    safe_add(gt_hr_fft_all, gt_fft_hr)
                    safe_add(predict_hr_fft_all, pre_fft_hr)
                    safe_add(SNR_all, SNR)
                    safe_add(MACC_all, macc)
                    
                    safe_add(gt_hr_temp, gt_fft_hr)
                    safe_add(pre_hr_temp, pre_fft_hr)
                    safe_add(SNR_temp, SNR)
                    safe_add(macc_temp, macc)
                else:
                    raise ValueError("Inference evaluation method name wrong!")
            
            if config.INFERENCE.SAVE_BVP:
                np.save(f"{config.INFERENCE.SAVE_PATH}/{method_name}/gt_hr_{i_batch}_{idx}.npy", np.array(gt_hr_temp))
                np.save(f"{config.INFERENCE.SAVE_PATH}/{method_name}/pre_hr_{i_batch}_{idx}.npy", np.array(pre_hr_temp))
                np.save(f"{config.INFERENCE.SAVE_PATH}/{method_name}/SNR_{i_batch}_{idx}.npy", np.array(SNR_temp))
                np.save(f"{config.INFERENCE.SAVE_PATH}/{method_name}/macc_{i_batch}_{idx}.npy", np.array(macc_temp))
            
    print("Used Unsupervised Method: " + method_name)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]', 
                    y_label='Average of rPPG HR and GT PPG HR [bpm]', 
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
            
    else:
        raise ValueError("Inference evaluation method name wrong!")
