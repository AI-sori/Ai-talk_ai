import cv2
import numpy as np
from .pupil_detector import PupilDetector

class SimpleGazeEstimator:
    """개선된 시선 방향 추정"""
    
    def __init__(self):
        self.pupil_detector = PupilDetector()
        
        # **임계값 조정** (더 민감하게)
        self.left_threshold = -0.1   # 더 작게
        self.right_threshold = 0.1   # 더 작게
    
    def estimate_direction(self, eye_image):
        """시선 방향 추정 - 개선 버전"""
        pupil = self.pupil_detector.detect_pupil(eye_image)
        
        if not pupil:
            return None
        
        h, w = eye_image.shape[:2]
        eye_center_x = w / 2
        
        pupil_x = pupil[0]
        
        # 정규화된 오프셋
        offset = pupil_x - eye_center_x
        offset_ratio = offset / eye_center_x
        
        print(f"[DEBUG] 동공X={pupil_x:.1f}, 중심X={eye_center_x:.1f}, 비율={offset_ratio:.3f}")
        
        # 방향 분류
        if offset_ratio < self.left_threshold:
            direction = 'left'
        elif offset_ratio > self.right_threshold:
            direction = 'right'
        else:
            direction = 'center'
        
        return direction
    
    def estimate_gaze_vector(self, eye_image):
        """시선 벡터"""
        pupil = self.pupil_detector.detect_pupil(eye_image)
        
        if not pupil:
            return None
        
        h, w = eye_image.shape[:2]
        
        gaze_x = (pupil[0] / w - 0.5) * 2
        gaze_y = (pupil[1] / h - 0.5) * 2
        
        return [gaze_x, gaze_y]
    
    def visualize(self, eye_image):
        """시각화"""
        direction = self.estimate_direction(eye_image)
        gaze_vector = self.estimate_gaze_vector(eye_image)
        
        # 동공 표시
        vis, pupil = self.pupil_detector.visualize(eye_image)
        
        # 방향 텍스트
        if direction:
            color = {
                'left': (255, 0, 0),    # 파랑
                'center': (0, 255, 0),  # 초록
                'right': (0, 0, 255)    # 빨강
            }[direction]
            
            cv2.putText(
                vis, direction.upper(),
                (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )
        
        return vis, direction, gaze_vector


# 테스트
# gaze_estimator.py 맨 아래 테스트 부분만 교체

if __name__ == "__main__":
    estimator = SimpleGazeEstimator()
    cap = cv2.VideoCapture(0)
    
    print("=== 시선 방향 테스트 ===")
    print("얼굴을 화면 중앙에 위치시키고")
    print("왼쪽 눈만 사용합니다")
    print("q로 종료")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # ✅ 왼쪽 눈 영역만 정확히 추출
        # 비율 조정: 눈 위치에 맞게
        left_eye = frame[
            int(h * 0.35):int(h * 0.50),  # 세로: 상단 35%~50%
            int(w * 0.25):int(w * 0.42)   # 가로: 좌측 25%~42%
        ]
        
        # 눈 영역이 너무 작으면 스킵
        if left_eye.shape[0] < 30 or left_eye.shape[1] < 30:
            print("눈 영역이 너무 작습니다")
            continue
        
        # 시선 추정
        vis, direction, gaze = estimator.visualize(left_eye)
        
        # 원본 프레임에 눈 영역 표시
        cv2.rectangle(
            frame,
            (int(w * 0.25), int(h * 0.35)),
            (int(w * 0.42), int(h * 0.50)),
            (0, 255, 0), 2
        )
        
        # 방향 텍스트를 원본 프레임에
        if direction:
            color = {
                'left': (255, 0, 0),
                'center': (0, 255, 0),
                'right': (0, 0, 255)
            }[direction]
            
            cv2.putText(
                frame, 
                f"Gaze: {direction.upper()}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3
            )
        
        # 두 화면 모두 표시
        cv2.imshow('Full Frame', frame)
        cv2.imshow('Left Eye Only', vis)
        
        if direction:
            print(f"방향: {direction:6s}, 벡터: {gaze}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()