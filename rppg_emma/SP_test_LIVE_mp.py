from turtle import left
import numpy as np
from scipy.spatial import ConvexHull
import cv2
from PIL import Image, ImageDraw
from numba import njit, prange, float32
import mediapipe as mp
from scipy.signal import welch
from multiprocessing import Process, Queue

class SignalProcessingParams():
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)

class SkinProcessingParams():
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)



def cpu_OMIT(signal):
    """
    OMIT method on CPU using Numpy.

    Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
    """
    X = signal
    Q, R = np.linalg.qr(X)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, X)
    bvp = Y[1, :]
    return bvp





def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax



class MagicLandmarks():
    """
    This class contains usefull lists of landmarks identification numbers.
    """
    # dense zones used for convex hull masks
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mouth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    nose = [193, 417, 168, 188, 6, 412, 197, 174, 399, 456, 195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305]

@njit(['float32[:,:](uint8[:,:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean


class SkinExtractionConvexHull:
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self):
        self.skin_mask = None #mask matrix multiplication results, mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mouth_mask)
    
    def extract_skin(self,image, ldmks, use_past_mask):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].
            use_past_mask(boolean): use past frame`s mask to reduce the amount of calculation
        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        if not use_past_mask:
            aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
            # face_mask convex hull 
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mask = np.array(img)
            mask = np.expand_dims(mask,axis=0).T

            # left eye convex hull
            left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
            aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
            if len(aviable_ldmks) > 3:
                hull = ConvexHull(aviable_ldmks)
                verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
                img = Image.new('L', image.shape[:2], 0)
                ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
                left_eye_mask = np.array(img)
                left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
            else:
                left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

            # right eye convex hull
            right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
            aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
            if len(aviable_ldmks) > 3:
                hull = ConvexHull(aviable_ldmks)
                verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
                img = Image.new('L', image.shape[:2], 0)
                ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
                right_eye_mask = np.array(img)
                right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
            else:
                right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

            # mouth convex hull
            mouth_ldmks = ldmks[MagicLandmarks.mouth]
            aviable_ldmks = mouth_ldmks[mouth_ldmks[:,0] >= 0][:,:2]
            if len(aviable_ldmks) > 3:
                hull = ConvexHull(aviable_ldmks)
                verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
                img = Image.new('L', image.shape[:2], 0)
                ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
                mouth_mask = np.array(img)
                mouth_mask = np.expand_dims(mouth_mask,axis=0).T
            else:
                mouth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)
                
            # nose convex hull
            nose_ldmks = ldmks[MagicLandmarks.nose]
            aviable_ldmks = nose_ldmks[nose_ldmks[:,0] >= 0][:,:2]
            if len(aviable_ldmks) > 3:
                hull = ConvexHull(aviable_ldmks)
                verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
                img = Image.new('L', image.shape[:2], 0)
                ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
                nose_mask = np.array(img)
                nose_mask = np.expand_dims(nose_mask,axis=0).T
            else:
                nose_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)
                
                
            #save mask to use later
            self.skin_mask = mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mouth_mask) * (1-nose_mask)

        skin_image = image * self.skin_mask
        rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        cropped_skin_im = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

        return cropped_skin_im



class SignalProcessing():
    def __init__(self, frame_queue):
        # Common parameters #
        self.skin_extractor = SkinExtractionConvexHull()
        self.frame_queue = frame_queue

    ### HOLISTIC METHODS ###
    def extract_holistic(self, face_detection_interval=60): #1 face detection per face_detection_interval frame
        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        #processed_frames_count = 0
        frame_pass_count = 0
        ldmks = None
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while True:
                frame = self.frame_queue.get(timeout=5) #2초간 못받으면 종료

                frame_pass_count+=1
                if frame_pass_count%face_detection_interval != 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                    if ldmks is not None:
                        cropped_skin_im = skin_ex.extract_skin(image, ldmks, use_past_mask=True)
                    else:
                        cropped_skin_im = np.zeros_like(image)
                else:
                    # convert the BGR image to RGB.
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    width = image.shape[1]
                    height = image.shape[0]
                    # [landmarks, info], with info->x_center ,y_center, r, g, bf
                    ldmks = np.zeros((468, 5), dtype=np.float32)
                    ldmks[:, 0] = -1.0
                    ldmks[:, 1] = -1.0
                    ### face landmarks ###
                    results = face_mesh.process(image)
                    if results.multi_face_landmarks: #try except가 더 빠를지도. 정상적으로 얼굴 찾는 경우가 많을테니.
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = [l for l in face_landmarks.landmark]
                        for idx in range(len(landmarks)):
                            landmark = landmarks[idx]
                            if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                                coords = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                                if coords:
                                    ldmks[idx, 0] = coords[1]
                                    ldmks[idx, 1] = coords[0]
                        ### skin extraction ###
                        cropped_skin_im = skin_ex.extract_skin(image, ldmks, use_past_mask=False)
                    else:
                        cropped_skin_im = np.zeros_like(image)
                        print("Face is not detected.")
                        ldmks = None

                #cv2.imshow('cm',cropped_skin_im)
                #cv2.waitKey(1)
                    ### sig computing ###
                RGB_mean = holistic_mean(cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)) # shape:(1,3)
                yield RGB_mean
                    





def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(float32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

class BPM:
    """
    Provides BPMs estimate from BVP signals using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz


    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        #Normalized Power에서 획득하는 SNR은, 일반 SNR과 비교하면 min(Power) 값이 penalty term 역할을 함.
        Pmax = np.argmax(Power, axis=1)  # power max
        SNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        Power = (Power-np.min(Power))/(np.max(Power)-np.min(Power))
        pSNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        return Pfreqs[Pmax.squeeze()], SNR, pSNR, Pfreqs, Power