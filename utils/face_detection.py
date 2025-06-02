import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # MediaPipe 설정 - 매우 민감하게
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1,  # 매우 낮춤 - 더 민감
            min_tracking_confidence=0.1    # 매우 낮춤 - 더 민감
        )
        
        # 눈 영역 랜드마크 인덱스 (MediaPipe FaceMesh)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    def extract_eyes(self, frame):
        """프레임에서 눈 영역 추출"""
        try:
            print(f"[DEBUG] 입력 프레임 크기: {frame.shape}")
            print(f"[DEBUG] 프레임 타입: {frame.dtype}, 최대값: {frame.max()}, 최소값: {frame.min()}")
            
            # 이미지 저장해서 확인 (디버깅용)
            cv2.imwrite('debug_frame.jpg', frame)
            print("[DEBUG] 이미지를 debug_frame.jpg로 저장했습니다")
            
            # 이미지 전처리 개선
            if frame.shape[0] < 480 or frame.shape[1] < 640:
                # 너무 작으면 크기 조정
                frame = cv2.resize(frame, (640, 480))
                print(f"[DEBUG] 프레임 크기 조정됨: {frame.shape}")
            
            # OpenCV 얼굴 검출로 먼저 테스트
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            print(f"[DEBUG] OpenCV로 검출된 얼굴 개수: {len(faces)}")
            
            if len(faces) == 0:
                print("[DEBUG] OpenCV도 얼굴을 찾지 못했습니다 - 이미지 품질 문제일 수 있습니다")
                return None, None, None
            else:
                print(f"[DEBUG] OpenCV는 얼굴을 찾았습니다: {faces}")
            
            # MediaPipe 시도
            # RGB 변환 (MediaPipe는 RGB 필요)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[DEBUG] RGB 변환 완료: {rgb_frame.shape}")
            
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                print("[DEBUG] MediaPipe에서 얼굴을 찾지 못했습니다")
                
                # MediaPipe 설정을 더 관대하게 변경
                print("[DEBUG] MediaPipe 설정을 더 관대하게 변경...")
                face_mesh_lenient = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,  # static 모드로 변경
                    max_num_faces=1,
                    refine_landmarks=False,  # refine 끄기
                    min_detection_confidence=0.05,  # 더 낮춤
                    min_tracking_confidence=0.05
                )
                
                results = face_mesh_lenient.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    print("[DEBUG] 관대한 설정으로도 실패")
                    return None, None, None
                else:
                    print("[DEBUG] 관대한 설정으로 성공!")
            
            print(f"[DEBUG] 얼굴 {len(results.multi_face_landmarks)}개 검출됨")
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # 랜드마크를 픽셀 좌표로 변환
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # 왼쪽 눈 영역 추출
            left_eye = self._extract_eye_region(frame, landmarks, self.left_eye_indices)
            print(f"[DEBUG] 왼쪽 눈 크기: {left_eye.shape if left_eye is not None else 'None'}")
            
            # 오른쪽 눈 영역 추출
            right_eye = self._extract_eye_region(frame, landmarks, self.right_eye_indices)
            print(f"[DEBUG] 오른쪽 눈 크기: {right_eye.shape if right_eye is not None else 'None'}")
            
            # 얼굴 중심점 계산
            face_center = self._get_face_center(landmarks)
            print(f"[DEBUG] 얼굴 중심: {face_center}")
            
            return left_eye, right_eye, face_center
            
        except Exception as e:
            print(f"Error extracting eyes: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _extract_eye_region(self, frame, landmarks, eye_indices):
        """특정 눈 영역 추출"""
        try:
            # 눈 영역 좌표 추출
            eye_points = [landmarks[i] for i in eye_indices]
            
            # 바운딩 박스 계산
            x_coords = [p[0] for p in eye_points]
            y_coords = [p[1] for p in eye_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # 여백 추가
            margin = 10
            min_x = max(0, min_x - margin)
            min_y = max(0, min_y - margin)
            max_x = min(frame.shape[1], max_x + margin)
            max_y = min(frame.shape[0], max_y + margin)
            
            # 눈 영역 크롭
            eye_region = frame[min_y:max_y, min_x:max_x]
            
            # 최소 크기 확인
            if eye_region.shape[0] < 20 or eye_region.shape[1] < 20:
                return None
            
            return eye_region
            
        except Exception as e:
            print(f"Error extracting eye region: {e}")
            return None
    
    def _get_face_center(self, landmarks):
        """얼굴 중심점 계산"""
        try:
            # 얼굴 주요 점들의 평균으로 중심점 계산
            key_points = [landmarks[i] for i in [10, 152, 234, 454]]  # 이마, 턱, 양 볼
            center_x = sum(p[0] for p in key_points) // len(key_points)
            center_y = sum(p[1] for p in key_points) // len(key_points)
            return (center_x, center_y)
        except:
            return (0, 0)
    
    def draw_landmarks(self, frame, show_eyes=True):
        """랜드마크 시각화 (디버깅용)"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                if show_eyes:
                    # 눈 영역 표시
                    for idx in self.left_eye_indices + self.right_eye_indices:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            return frame
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
            return frame