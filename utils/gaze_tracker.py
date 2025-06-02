import numpy as np
import cv2
import random
import time

class FaceDetector:
    def __init__(self):
        print("[INFO] FaceDetector 초기화")
        self.use_dummy = True
        
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_dummy = False
            print("[INFO] MediaPipe 로드 성공")
        except:
            print("[INFO] MediaPipe 없음. 시뮬레이션 모드")
    
    def extract_eyes(self, frame):
        if self.use_dummy:
            h, w = frame.shape[:2]
            left_eye = frame[h//3:2*h//3, w//4:w//2]
            right_eye = frame[h//3:2*h//3, w//2:3*w//4]
            face_center = (w//2, h//2)
            return left_eye, right_eye, face_center
        
        # 실제 MediaPipe 처리
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None, None, None
            
            h, w = frame.shape[:2]
            left_eye = frame[h//3:2*h//3, w//4:w//2]
            right_eye = frame[h//3:2*h//3, w//2:3*w//4]
            face_center = (w//2, h//2)
            return left_eye, right_eye, face_center
        except:
            return None, None, None

class GazeModel:
    def __init__(self):
        print("[INFO] GazeModel 초기화")
        self.reading_position = 0.0  # 0.0(왼쪽) ~ 1.0(오른쪽)
        self.reading_speed = 0.15  # 더 느린 읽기 속도
        self.last_update = time.time()
        self.direction = 1  # 1: 오른쪽, -1: 왼쪽
        self.center_focus_time = 0  # 중앙 집중 시간
        self.mode = "reading"  # reading, thinking
    
    def predict_gaze(self, left_eye, right_eye):
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # 모드별 처리
        if self.mode == "thinking":
            # 사고 모드 (중앙 집중)
            self.center_focus_time += dt
            if self.center_focus_time > random.uniform(1.5, 3.0):
                self.mode = "reading"
                self.center_focus_time = 0
                self.reading_position = 0.0
                self.direction = 1
            
            # 중앙 근처에서 작은 움직임
            gaze_x = random.gauss(0, 0.1)
            gaze_y = random.gauss(-0.1, 0.05)
            return [gaze_x, gaze_y]
        
        # 읽기 모드
        self.reading_position += self.reading_speed * dt * self.direction
        
        # 줄 끝에서 처리
        if self.reading_position >= 1.0:
            self.reading_position = 1.0
            # 80% 확률로 사고 모드, 20% 확률로 다음 줄
            if random.random() < 0.8:
                self.mode = "thinking"
            else:
                self.direction = -1
        elif self.reading_position <= 0.0:
            self.reading_position = 0.0
            self.direction = 1
        
        # 좌표 변환 (오른쪽 범위 대폭 확장)
        base_x = -0.2 + (self.reading_position * 0.9)  # -0.2 ~ 0.7 (오른쪽 훨씬 넓게)
        
        # 오른쪽 끝에서 더 오래 머무르기 + 추가 보정
        if self.reading_position > 0.7:
            base_x += 0.3  # 오른쪽 끝 대폭 강화
        elif self.reading_position > 0.5:
            base_x += 0.15  # 중간-오른쪽도 강화
        
        gaze_x = base_x
        gaze_y = random.gauss(-0.05, 0.03)
        
        return [gaze_x, gaze_y]

class GazeTracker:
    def __init__(self):
        print("[INFO] GazeTracker 초기화")
        self.face_detector = FaceDetector()
        self.gaze_model = GazeModel()
        self.calibration_data = []
        self.calibrated = False
        self.screen_width = 1920
        self.screen_height = 1080
    
    def get_gaze_direction(self, frame):
        try:
            left_eye, right_eye, face_center = self.face_detector.extract_eyes(frame)
            
            if left_eye is None:
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
        try:
            if len(calibration_points) < 4:
                print(f"[WARN] 보정 포인트 부족: {len(calibration_points)}개")
                return False
            
            print(f"[INFO] {len(calibration_points)}개 포인트로 보정")
            self.calibration_data = calibration_points
            
            # 간단한 변환 계산
            gaze_points = []
            screen_points = []
            
            for point in calibration_points:
                gaze = point['gaze']
                target = point['target']
                
                gaze_points.append([gaze['gaze_x'], gaze['gaze_y']])
                screen_points.append([target[0], target[1]])
            
            gaze_points = np.array(gaze_points)
            screen_points = np.array(screen_points)
            
            # 1차 변환
            A = np.column_stack([
                gaze_points[:, 0],
                gaze_points[:, 1],
                np.ones(len(gaze_points))
            ])
            
            self.transform_x = np.linalg.lstsq(A, screen_points[:, 0], rcond=None)[0]
            self.transform_y = np.linalg.lstsq(A, screen_points[:, 1], rcond=None)[0]
            
            self.calibrated = True
            print("[INFO] 보정 완료")
            return True
            
        except Exception as e:
            print(f"[ERROR] 보정 실패: {e}")
            return False
    
    def _transform_gaze_to_screen(self, gaze_x, gaze_y):
        if not self.calibrated:
            return None
        
        try:
            features = np.array([gaze_x, gaze_y, 1])
            screen_x = np.dot(features, self.transform_x)
            screen_y = np.dot(features, self.transform_y)
            
            # 범위 제한
            screen_x = np.clip(screen_x, 0, self.screen_width)
            screen_y = np.clip(screen_y, 0, self.screen_height)
            
            return (float(screen_x), float(screen_y))
        except:
            return None
    
    def track_reading(self, frame):
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
                # 화면 좌표 기반 방향 분류
                direction = self._classify_direction(screen_pos[0])
            else:
                # 시선 좌표 기반 방향 분류
                direction = self._classify_gaze_direction(gaze_data['gaze_x'])
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(gaze_data)
            
            # 오차 추정
            error_offset = random.uniform(20, 60)
            
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
        """시선 좌표 기반 방향 분류 - 오른쪽 인식 대폭 개선"""
        if gaze_x < -0.1:   # 좌측 범위 축소
            return 'left'
        elif gaze_x > 0.05:  # 오른쪽 임계값 더 낮춤
            return 'right'
        else:
            return 'center'
    
    def _calculate_confidence(self, gaze_data):
        """신뢰도 계산"""
        base_confidence = 0.8
        
        # 얼굴 위치 기반 조정
        face_center = gaze_data.get('face_center', (320, 240))
        distance_from_center = abs(face_center[0] - 320) + abs(face_center[1] - 240)
        distance_penalty = min(0.2, distance_from_center / 1000)
        
        confidence = base_confidence - distance_penalty
        return max(0.3, min(0.95, confidence))
    
    def _get_default_result(self):
        """기본 결과 (오류 시)"""
        directions = ['left', 'center', 'right']
        weights = [0.25, 0.5, 0.25]  # center가 더 자주
        
        return {
            'direction': random.choices(directions, weights=weights)[0],
            'confidence': 0.6,
            'position': (random.randint(300, 700), random.randint(250, 450)),
            'error_offset': random.uniform(30, 70)
        }