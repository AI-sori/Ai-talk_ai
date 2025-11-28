# face_gaze_tracker.py 새로 만들기

import cv2
import mediapipe as mp
from gaze_estimator import SimpleGazeEstimator

class FaceGazeTracker:
    """얼굴 검출 + 시선 추적 통합"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 시선 추정기
        self.gaze_estimator = SimpleGazeEstimator()
        
        # 눈 영역 랜드마크 인덱스
        self.left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 153]
        self.right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 373]

        #임계값 조정
        self.left_threshold = -0.03   # 더 좁게!
        self.right_threshold = 0.03   # 더 좁게!

    def estimate_direction_from_pupil(self, eye_image):
        """동공 위치로 방향 추정 - 개선"""
        if eye_image is None:
            return None

        pupil = self.pupil_detector.detect_pupil(eye_image)

        if not pupil:
            return None

        h, w = eye_image.shape[:2]
        eye_center_x = w / 2

        pupil_x = pupil[0]
        offset = pupil_x - eye_center_x
        offset_ratio = offset / eye_center_x

        # ✅ 디버깅용 출력 (임계값 조정에 참고)
        # print(f"[DEBUG] offset_ratio: {offset_ratio:.3f}")

        # ✅ 더 민감한 분류
        if offset_ratio < self.left_threshold:
            return 'left'
        elif offset_ratio > self.right_threshold:
            return 'right'
        else:
            return 'center'

    def extract_eye_region(self, frame, landmarks, eye_indices):
        """랜드마크로 눈 영역 추출"""
        h, w = frame.shape[:2]
        
        # 눈 랜드마크 좌표
        points = []
        for idx in eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        # 바운딩 박스
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 여백 추가
        margin = 10
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(w, max_x + margin)
        max_y = min(h, max_y + margin)
        
        # 눈 영역 크롭
        eye_region = frame[min_y:max_y, min_x:max_x]
        
        return eye_region, (min_x, min_y)
    
    def track(self, frame):
        """
        프레임에서 시선 추적
        Returns: {
            'left_direction': 'left'/'center'/'right',
            'right_direction': 'left'/'center'/'right',
            'final_direction': 'left'/'center'/'right',
            'gaze_vector': [x, y],
            'face_detected': True/False
        }
        """
        # RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'face_detected': False,
                'final_direction': None
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 왼쪽 눈
        left_eye, left_pos = self.extract_eye_region(
            frame, landmarks, self.left_eye_indices
        )
        left_dir = self.gaze_estimator.estimate_direction(left_eye)
        left_gaze = self.gaze_estimator.estimate_gaze_vector(left_eye)
        
        # 오른쪽 눈
        right_eye, right_pos = self.extract_eye_region(
            frame, landmarks, self.right_eye_indices
        )
        right_dir = self.gaze_estimator.estimate_direction(right_eye)
        right_gaze = self.gaze_estimator.estimate_gaze_vector(right_eye)
        
        # 최종 방향 (양쪽 눈 평균)
        if left_dir and right_dir:
            if left_dir == right_dir:
                final_dir = left_dir
            else:
                # 불일치 시 center로
                final_dir = 'center'
        else:
            final_dir = left_dir or right_dir
        
        # 시선 벡터 평균
        if left_gaze and right_gaze:
            avg_gaze = [
                (left_gaze[0] + right_gaze[0]) / 2,
                (left_gaze[1] + right_gaze[1]) / 2
            ]
        else:
            avg_gaze = left_gaze or right_gaze
        
        return {
            'face_detected': True,
            'left_direction': left_dir,
            'right_direction': right_dir,
            'final_direction': final_dir,
            'gaze_vector': avg_gaze,
            'left_eye_region': (left_eye, left_pos),
            'right_eye_region': (right_eye, right_pos)
        }


# 테스트
if __name__ == "__main__":
    tracker = FaceGazeTracker()
    cap = cv2.VideoCapture(0)
    
    print("통합 시선 추적 테스트 (q로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ 좌우 반전 (자연스럽게)
        frame = cv2.flip(frame, 1)
        
        # 추적
        result = tracker.track(frame)
        
        # 결과 표시
        if result['face_detected']:
            direction = result['final_direction']
            
            # 방향 텍스트
            color = {
                'left': (255, 0, 0),
                'center': (0, 255, 0),
                'right': (0, 0, 255)
            }.get(direction, (255, 255, 255))
            
            cv2.putText(
                frame, 
                f"Gaze: {direction or 'Unknown'}", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, color, 3
            )
            
            # 눈 영역 표시
            left_eye, left_pos = result['left_eye_region']
            right_eye, right_pos = result['right_eye_region']
            
            cv2.rectangle(
                frame,
                left_pos,
                (left_pos[0] + left_eye.shape[1], 
                 left_pos[1] + left_eye.shape[0]),
                (0, 255, 0), 2
            )
            
            print(f"방향: {direction}, 벡터: {result['gaze_vector']}")
        else:
            cv2.putText(
                frame, "No Face Detected", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )
        
        cv2.imshow('Face Gaze Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()