try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe가 설치되어 있지 않습니다. pip install mediapipe로 설치하세요.")
try:
    import numpy as np
except ImportError:
    raise ImportError("numpy가 설치되어 있지 않습니다. pip install numpy로 설치하세요.")
try:
    import cv2
except ImportError:
    raise ImportError("opencv-python이 설치되어 있지 않습니다. pip install opencv-python으로 설치하세요.")

class mp_facemesh:
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    @staticmethod
    def get_face_bbox(frame):
        # mediapipe face detection으로 bounding box 추출
        results = mp_facemesh.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            det = results.detections[0]
            bboxC = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            return [x, y, w, h]
        else:
            # 얼굴이 없으면 전체 프레임 반환
            return [0, 0, frame.shape[1], frame.shape[0]]

    @staticmethod
    def get_skin_mask(frame):
        # mediapipe face mesh로 얼굴 랜드마크 기반 마스크 생성
        results = mp_facemesh.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            # 볼, 이마 등 주요 피부 영역 인덱스(예시: 10, 338, 297, 332, 284, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
            # 실제 사용할 영역은 프로젝트 목적에 맞게 조정 가능
            skin_idx = [10, 338, 297, 332, 284, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            for face_landmarks in results.multi_face_landmarks:
                points = np.array([(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in skin_idx])
                if len(points) > 0:
                    cv2.fillPoly(mask, [points], 255)
        return mask