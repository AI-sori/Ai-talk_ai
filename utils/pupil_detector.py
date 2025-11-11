import cv2
import numpy as np

class PupilDetector:
    """개선된 동공 검출기"""
    
    def detect_pupil(self, eye_image):
        """동공 중심 찾기 - 개선 버전"""
        try:
            if eye_image is None or eye_image.size == 0:
                return None
            
            # 1. 그레이스케일 변환
            if len(eye_image.shape) == 3:
                gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_image
            
            # 2. 히스토그램 평활화 (조명 보정)
            gray = cv2.equalizeHist(gray)
            
            # 3. 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 4. **핵심 수정**: 가장 어두운 영역 찾기
            # Otsu 대신 직접 최소값 찾기
            min_val = blurred.min()
            
            # 최소값 근처만 선택 (동공은 가장 어두움)
            threshold_val = min_val + 20  # 여유값
            _, binary = cv2.threshold(
                blurred, threshold_val, 255, cv2.THRESH_BINARY_INV
            )
            
            # 5. 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 6. 컨투어 찾기
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # 7. 동공 후보 필터링
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # 면적 필터 (너무 작거나 크면 제외)
                if area < 20 or area > (w * h * 0.4):
                    continue
                
                # 중심에 가까운 것 우선
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # 눈 영역 중심 근처인지 확인
                dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                valid_contours.append({
                    'contour': cnt,
                    'center': (cx, cy),
                    'area': area,
                    'dist': dist_to_center
                })
            
            if not valid_contours:
                return None
            
            # 8. 중심에 가장 가까운 것 선택
            best = min(valid_contours, key=lambda x: x['dist'])
            
            return best['center']
            
        except Exception as e:
            print(f"동공 검출 오류: {e}")
            return None
    
    def visualize(self, eye_image, show_process=False):
        """시각화 + 디버깅 정보"""
        pupil = self.detect_pupil(eye_image)
        
        vis = eye_image.copy()
        
        if pupil:
            # 동공 표시
            cv2.circle(vis, pupil, 3, (0, 0, 255), -1)  # 빨간 점
            cv2.circle(vis, pupil, 8, (0, 255, 0), 2)   # 초록 원
            
            # 좌표 표시
            cv2.putText(
                vis, f"({pupil[0]}, {pupil[1]})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
        else:
            cv2.putText(
                vis, "No Pupil",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )
        
        # 중심선 표시
        h, w = vis.shape[:2]
        cv2.line(vis, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        
        return vis, pupil


# 테스트
if __name__ == "__main__":
    detector = PupilDetector()
    cap = cv2.VideoCapture(0)
    
    print("=== 동공 검출 테스트 ===")
    print("키 안내:")
    print("  q: 종료")
    print("  s: 현재 프레임 저장")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 눈 영역 (임시)
        h, w = frame.shape[:2]
        eye_region = frame[h//3:2*h//3, w//4:3*w//4]
        
        # 동공 검출
        vis, pupil = detector.visualize(eye_region)
        
        # 화면 표시
        cv2.imshow('Pupil Detection Test', vis)
        
        if pupil:
            print(f"Frame {frame_count}: 동공 위치 = {pupil}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'debug_frame_{frame_count}.jpg', vis)
            print(f"프레임 저장: debug_frame_{frame_count}.jpg")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()