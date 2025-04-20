import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # OpenCV의 얼굴 및 눈 검출기 로드
        haar_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(haar_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(haar_path + 'haarcascade_eye.xml')
        
        # 감지기 로드 확인
        if self.face_cascade.empty():
            raise ValueError("Face cascade failed to load")
        if self.eye_cascade.empty():
            raise ValueError("Eye cascade failed to load")
    
    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # 첫 번째 얼굴 사용
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # 눈 감지 (예외 처리 추가)
        try:
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            if len(eyes) == 0:
                return faces[0], []  # 눈이 감지되지 않으면 빈 리스트 반환
        except:
            return faces[0], []  # 예외 발생 시 빈 리스트 반환
        
        return faces[0], eyes
    
    def extract_eyes(self, frame, face_eyes):
        face, eyes = face_eyes
        if face is None or len(eyes) < 2:
            return None, None, None, None
        
        (fx, fy, fw, fh) = face
        
        # 눈이 2개보다 많이 감지될 경우, 가장 큰 두 개만 선택
        if len(eyes) > 2:
            # 크기(너비*높이) 기준으로 정렬
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        
        # 왼쪽과 오른쪽 눈 구분 (y 좌표가 비슷하고 x 좌표로 구분)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # 눈 위치에 마진 추가 (더 큰 눈 영역)
        margin = 5
        
        # 눈 좌표 계산 (전체 프레임 기준)
        left_eye_rect = (
            max(0, fx + eyes[0][0] - margin), 
            max(0, fy + eyes[0][1] - margin), 
            min(frame.shape[1], fx + eyes[0][0] + eyes[0][2] + margin), 
            min(frame.shape[0], fy + eyes[0][1] + eyes[0][3] + margin)
        )
        
        right_eye_rect = (
            max(0, fx + eyes[1][0] - margin), 
            max(0, fy + eyes[1][1] - margin), 
            min(frame.shape[1], fx + eyes[1][0] + eyes[1][2] + margin), 
            min(frame.shape[0], fy + eyes[1][1] + eyes[1][3] + margin)
        )
        
        # 눈 이미지 추출
        left_eye = frame[left_eye_rect[1]:left_eye_rect[3], left_eye_rect[0]:left_eye_rect[2]]
        right_eye = frame[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]
        
        return left_eye, right_eye, left_eye_rect, right_eye_rect