# pupil_detector.py 새로 만들기

import cv2
import numpy as np

class PupilDetector:
    """간단한 동공 검출기"""
    
    def detect_pupil(self, eye_image):
        """
        눈 이미지에서 동공 중심 찾기
        Returns: (cx, cy) 또는 None
        """
        try:
            # 1. 그레이스케일 변환
            if len(eye_image.shape) == 3:
                gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_image
            
            # 2. 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # 3. 임계값으로 동공 추출 (어두운 부분)
            _, threshold = cv2.threshold(
                blurred, 60, 255, cv2.THRESH_BINARY_INV
            )
            
            # 4. 컨투어 찾기
            contours, _ = cv2.findContours(
                threshold, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # 5. 가장 큰 컨투어 = 동공
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 면적이 너무 작으면 무시
            if cv2.contourArea(largest_contour) < 20:
                return None
            
            # 6. 모멘트로 중심 계산
            M = cv2.moments(largest_contour)
            if M['m00'] == 0:
                return None
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            return (cx, cy)
            
        except Exception as e:
            print(f"동공 검출 오류: {e}")
            return None
    
    def visualize(self, eye_image):
        """동공 검출 과정 시각화 (디버깅용)"""
        pupil = self.detect_pupil(eye_image)
        
        # 원본 이미지에 표시
        vis = eye_image.copy()
        
        if pupil:
            cv2.circle(vis, pupil, 3, (0, 0, 255), -1)  # 빨간 점
            cv2.circle(vis, pupil, 10, (0, 255, 0), 2)  # 초록 원
        
        return vis, pupil

# 테스트 코드
if __name__ == "__main__":
    detector = PupilDetector()
    
    # 웹캠으로 테스트
    cap = cv2.VideoCapture(0)
    print("동공 검출 테스트 (q로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 간단히 화면 중앙 영역을 눈이라고 가정
        h, w = frame.shape[:2]
        eye_region = frame[h//3:2*h//3, w//4:3*w//4]
        
        # 동공 검출
        vis, pupil = detector.visualize(eye_region)
        
        # 결과 표시
        cv2.imshow('Eye Region', vis)
        
        if pupil:
            print(f"동공 위치: {pupil}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()