# utils/gaze_tracker.py (완전 새로 작성)

import numpy as np
import cv2
import mediapipe as mp
from .pupil_detector import PupilDetector  # 우리가 만든 것

class FaceDetector:
    """MediaPipe 기반 얼굴/눈 검출"""
    
    def __init__(self):
        print("[INFO] FaceDetector 초기화 (MediaPipe)")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 눈 랜드마크 인덱스
        self.left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 153]
        self.right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 373]
    
    def extract_eyes(self, frame):
        """
        프레임에서 양쪽 눈 영역 추출
        Returns: left_eye, right_eye, face_center
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None, None, None
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # 왼쪽 눈 추출
            left_eye = self._extract_eye_region(frame, landmarks, self.left_eye_indices)
            
            # 오른쪽 눈 추출
            right_eye = self._extract_eye_region(frame, landmarks, self.right_eye_indices)
            
            # 얼굴 중심 (코 끝)
            nose_tip = landmarks[1]
            face_center = (int(nose_tip.x * w), int(nose_tip.y * h))
            
            return left_eye, right_eye, face_center
            
        except Exception as e:
            print(f"[ERROR] 눈 추출 오류: {e}")
            return None, None, None
    
    def _extract_eye_region(self, frame, landmarks, eye_indices):
        """눈 영역 크롭"""
        h, w = frame.shape[:2]
        
        points = []
        for idx in eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 여백
        width = max_x - min_x
        height = max_y - min_y
        margin_x = int(width * 0.3)
        margin_y = int(height * 0.3)
        
        min_x = max(0, min_x - margin_x)
        min_y = max(0, min_y - margin_y)
        max_x = min(w, max_x + margin_x)
        max_y = min(h, max_y + margin_y)
        
        eye_region = frame[min_y:max_y, min_x:max_x]
        
        if eye_region.shape[0] < 20 or eye_region.shape[1] < 30:
            return None
        
        return eye_region


class GazeModel:
    """실제 동공 기반 시선 추정 모델"""
    
    def __init__(self):
        print("[INFO] GazeModel 초기화 (동공 기반)")
        self.pupil_detector = PupilDetector()
        
        # 임계값
        self.left_threshold = -0.08 #LEFT 범위 줄임
        self.right_threshold = 0.03 #RIGHT 범위 늘림
    
    def predict_gaze(self, left_eye, right_eye):
        """
        양쪽 눈에서 시선 방향 예측
        Returns: [gaze_x, gaze_y]
        """
        # 양쪽 눈의 동공 검출
        left_pupil = self.pupil_detector.detect_pupil(left_eye) if left_eye is not None else None
        right_pupil = self.pupil_detector.detect_pupil(right_eye) if right_eye is not None else None
        
        # 시선 벡터 계산
        gaze_vectors = []
        
        if left_pupil and left_eye is not None:
            h, w = left_eye.shape[:2]
            gaze_x = (left_pupil[0] / w - 0.5) * 2
            gaze_y = (left_pupil[1] / h - 0.5) * 2
            gaze_vectors.append([gaze_x, gaze_y])
        
        if right_pupil and right_eye is not None:
            h, w = right_eye.shape[:2]
            gaze_x = (right_pupil[0] / w - 0.5) * 2
            gaze_y = (right_pupil[1] / h - 0.5) * 2
            gaze_vectors.append([gaze_x, gaze_y])
        
        # 평균
        if gaze_vectors:
            avg_gaze = np.mean(gaze_vectors, axis=0)
            return avg_gaze.tolist()
        
        return [0.0, 0.0]  # 기본값


class GazeTracker:
    """통합 시선 추적 시스템 (기존 인터페이스 유지)"""
    
    def __init__(self):
        print("[INFO] GazeTracker 초기화")
        self.face_detector = FaceDetector()
        self.gaze_model = GazeModel()
        self.calibration_data = []
        self.calibrated = False
        self.screen_width = 1920
        self.screen_height = 1080
        
        # ✅ 평가용 추가
        self.calibration_errors = []  # 오차 기록
        self.avg_calibration_error = 0.0
    
    def get_gaze_direction(self, frame):
        """
        시선 좌표 추출 (기존 인터페이스 유지)
        Returns: {'gaze_x': float, 'gaze_y': float, 'face_center': tuple}
        """
        try:
            #화면 좌우반전추가
            frame = cv2.flip(frame, 1)
            left_eye, right_eye, face_center = self.face_detector.extract_eyes(frame)
            
            if left_eye is None and right_eye is None:
                return None
            
            gaze_pred = self.gaze_model.predict_gaze(left_eye, right_eye)
            
            return {
                'gaze_x': float(gaze_pred[0]),
                'gaze_y': float(gaze_pred[1]),
                'face_center': face_center or (320, 240)
            }
        except Exception as e:
            print(f"[ERROR] 시선 감지 오류: {e}")
            return None
    
    def calibrate(self, calibration_points):
        """
        보정 (기존 인터페이스 유지 + 오차 계산 추가)
        """
        try:
            if len(calibration_points) < 4:
                print(f"[WARN] 보정 포인트 부족: {len(calibration_points)}개")
                return False
            
            print(f"[INFO] {len(calibration_points)}개 포인트로 보정")
            self.calibration_data = calibration_points
            
            gaze_points = []
            screen_points = []
            
            for point in calibration_points:
                gaze = point['gaze']
                target = point['target']
                
                gaze_points.append([gaze['gaze_x'], gaze['gaze_y']])
                screen_points.append([target[0], target[1]])
            
            gaze_points = np.array(gaze_points)
            screen_points = np.array(screen_points)
            
            # 1차 변환 (기존과 동일)
            A = np.column_stack([
                gaze_points[:, 0],
                gaze_points[:, 1],
                np.ones(len(gaze_points))
            ])
            
            self.transform_x = np.linalg.lstsq(A, screen_points[:, 0], rcond=None)[0]
            self.transform_y = np.linalg.lstsq(A, screen_points[:, 1], rcond=None)[0]
            
            # ✅ 추가: 보정 오차 계산
            self.calibration_errors = []
            for point in calibration_points:
                predicted = self._transform_gaze_to_screen(
                    point['gaze']['gaze_x'],
                    point['gaze']['gaze_y']
                )
                actual = point['target']
                
                if predicted:
                    error = np.sqrt(
                        (predicted[0] - actual[0])**2 +
                        (predicted[1] - actual[1])**2
                    )
                    self.calibration_errors.append(error)
            
            if self.calibration_errors:
                self.avg_calibration_error = np.mean(self.calibration_errors)
                print(f"[INFO] ✅ 평균 보정 오차: {self.avg_calibration_error:.1f}px")
            
            self.calibrated = True
            print("[INFO] 보정 완료")
            return True
            
        except Exception as e:
            print(f"[ERROR] 보정 실패: {e}")
            return False
    
    def _transform_gaze_to_screen(self, gaze_x, gaze_y):
        """시선 좌표 → 화면 좌표 변환"""
        if not self.calibrated:
            return None
        
        try:
            features = np.array([gaze_x, gaze_y, 1])
            screen_x = np.dot(features, self.transform_x)
            screen_y = np.dot(features, self.transform_y)
            
            screen_x = np.clip(screen_x, 0, self.screen_width)
            screen_y = np.clip(screen_y, 0, self.screen_height)
            
            return (float(screen_x), float(screen_y))
        except:
            return None
    
    def track_reading(self, frame):
        """
        읽기 추적 (기존 인터페이스 유지)
        """
        try:
            gaze_data = self.get_gaze_direction(frame)
            if not gaze_data:
                return self._get_default_result()
            
            # 화면 좌표 변환
            screen_pos = self._transform_gaze_to_screen(
                gaze_data['gaze_x'],
                gaze_data['gaze_y']
            )
            
            if screen_pos:
                direction = self._classify_direction(screen_pos[0])
            else:
                direction = self._classify_gaze_direction(gaze_data['gaze_x'])
            
            confidence = self._calculate_confidence(gaze_data)
            
            # ✅ 수정: 실제 오차 사용
            error_offset = self.avg_calibration_error if self.calibrated else 50.0
            
            return {
                'direction': direction,
                'confidence': confidence,
                'position': screen_pos or (self.screen_width//2, self.screen_height//2),
                'error_offset': error_offset
            }
            
        except Exception as e:
            print(f"[ERROR] 추적 오류: {e}")
            return self._get_default_result()
    
    def _classify_direction(self, screen_x):
        """화면 좌표 기반 방향 분류"""
        left_boundary = self.screen_width * 0.4
        right_boundary = self.screen_width * 0.6
        
        if screen_x < left_boundary:
            return 'left'
        elif screen_x > right_boundary:
            return 'right'
        else:
            return 'center'
    
    def _classify_gaze_direction(self, gaze_x):
        """시선 좌표 기반 방향 분류"""
        if gaze_x < -0.05:
            return 'left'
        elif gaze_x > 0.05:
            return 'right'
        else:
            return 'center'
    
    def _calculate_confidence(self, gaze_data):
        """신뢰도 계산"""
        base_confidence = 0.8
        face_center = gaze_data.get('face_center', (320, 240))
        distance_from_center = abs(face_center[0] - 320) + abs(face_center[1] - 240)
        distance_penalty = min(0.2, distance_from_center / 1000)
        confidence = base_confidence - distance_penalty
        return max(0.3, min(0.95, confidence))
    
    def _get_default_result(self):
        """기본 결과"""
        return {
            'direction': 'center',
            'confidence': 0.6,
            'position': (self.screen_width//2, self.screen_height//2),
            'error_offset': 50.0
        }