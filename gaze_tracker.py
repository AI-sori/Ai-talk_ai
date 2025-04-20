import torch
import cv2
import numpy as np
from utils.face_detector import FaceDetector
from utils.eye_processor import EyeProcessor
from utils.gaze_visualizer import GazeVisualizer
from model_definition import GazeResNet

class GazeTracker:
    def __init__(self, model_path, use_cuda=True):
        # 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = GazeResNet(pretrained=False)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 체크포인트 구조 확인 및 로드
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print("모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("랜덤 초기화된 모델을 사용합니다.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 유틸리티 클래스 초기화
        self.face_detector = FaceDetector()
        self.eye_processor = EyeProcessor()
        self.visualizer = GazeVisualizer()
        
        # 성능 향상을 위한 시간적 평활화
        self.smoothing_window = 5
        self.gaze_history = []
    
    def smooth_gaze(self, gaze_vector):
        # 시간적 평활화 적용
        self.gaze_history.append(gaze_vector)
        if len(self.gaze_history) > self.smoothing_window:
            self.gaze_history.pop(0)
        
        # 최근 N개 벡터의 평균 계산
        avg_gaze = np.mean(self.gaze_history, axis=0)
        # 다시 단위 벡터로 정규화
        norm = np.linalg.norm(avg_gaze)
        if norm > 0:
            avg_gaze = avg_gaze / norm
        
        return avg_gaze
        
    
    def predict_gaze(self, eye_img):
        # 전처리
        eye_tensor = self.eye_processor.preprocess_eye(eye_img)
        if eye_tensor is None:
            return None
        
        # 추론
        with torch.no_grad():
            eye_tensor = eye_tensor.to(self.device)
            gaze_vector = self.model(eye_tensor).cpu().numpy()[0]
            print(f"Raw gaze vector: {gaze_vector}")  # 원시 출력 확인
    
        
        # 시간적 평활화 적용
        smoothed_gaze = self.smooth_gaze(gaze_vector)
        return smoothed_gaze

        return gaze_vector  # 원시 벡터 직접 반환
    
    def process_frame(self, frame):
        # 원본 프레임 복사
        result_frame = frame.copy()
        
        # 얼굴 및 눈 감지
        face, eyes = self.face_detector.detect_face(frame)
        if face is None:
            # 얼굴을 찾지 못함
            cv2.putText(result_frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame, None
        elif len(eyes) < 2:
            # 눈을 충분히 찾지 못함
            cv2.putText(result_frame, "Eyes not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame, None
        
        # 눈 영역 추출
        left_eye, right_eye, left_eye_rect, right_eye_rect = self.face_detector.extract_eyes(frame, (face, eyes))
        
        # 눈 이미지 추출 실패 확인
        if left_eye is None or right_eye is None or left_eye.size == 0 or right_eye.size == 0:
            cv2.putText(result_frame, "Eye extraction failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame, None
        
        # 시선 방향 예측 (오른쪽 눈 사용)
        raw_gaze_vector = self.predict_gaze(right_eye)

        # 예측 실패 시 왼쪽 눈으로 시도
        if raw_gaze_vector is None:
            raw_gaze_vector = self.predict_gaze(left_eye)

        # 두 눈 모두 예측 실패 시
        if raw_gaze_vector is None:
            cv2.putText(result_frame, "Gaze prediction failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame, None

        # 캘리브레이션 적용
        gaze_vector = self.apply_calibration(raw_gaze_vector)
        
        # 시각화
        # 눈 상자 그리기
        result_frame = self.visualizer.draw_eyes_bb(result_frame, left_eye_rect, right_eye_rect)
        
        # 시선 방향 화살표 그리기
        left_center = (
            (left_eye_rect[0] + left_eye_rect[2]) // 2,
            (left_eye_rect[1] + left_eye_rect[3]) // 2
        )
        right_center = (
            (right_eye_rect[0] + right_eye_rect[2]) // 2,
            (right_eye_rect[1] + right_eye_rect[3]) // 2
        )
        
        result_frame = self.visualizer.draw_gaze_direction(result_frame, left_center, gaze_vector)
        result_frame = self.visualizer.draw_gaze_direction(result_frame, right_center, gaze_vector)
        
        # 방향 텍스트 추가
        result_frame = self.visualizer.annotate_frame(result_frame, gaze_vector)
        
        return result_frame, gaze_vector
    
    # gaze_tracker.py에 캘리브레이션 기능 추가
    def calibrate(self, num_frames=30):
        """정면을 바라볼 때의 기준 벡터 설정"""
        print("정면을 바라보며 스페이스바를 누르세요...")
        
        cap = cv2.VideoCapture(0)
        calibration_vectors = []
        
        while len(calibration_vectors) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.putText(frame, f"Collected: {len(calibration_vectors)}/{num_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            
            key = cv2.waitKey(1)
            if key == 32:  # 스페이스바
                result_frame, gaze_vector = self.process_frame(frame)
                if gaze_vector is not None:
                    calibration_vectors.append(gaze_vector)
                    print(f"Captured vector: {gaze_vector}")
            
            if key == 27:  # ESC 키
                break
                
        cap.release()
        cv2.destroyWindow("Calibration")
        
        if calibration_vectors:
            self.reference_vector = np.mean(calibration_vectors, axis=0)
            print(f"Calibration complete. Reference vector: {self.reference_vector}")
            return True
        return False
    
    def calibrate_simple(self):
        """정면 바라볼 때의 기준 벡터 설정"""
        print("정면을 바라보고 스페이스바를 누르세요...")
        
        cap = cv2.VideoCapture(0)
        calibration_vectors = []
        
        cv2.namedWindow("Calibration")
        cv2.moveWindow("Calibration", 400, 200)
        
        # 보정점 이미지 생성
        calib_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.circle(calib_img, (300, 200), 10, (0, 0, 255), -1)
        cv2.circle(calib_img, (300, 200), 30, (0, 0, 255), 2)
        cv2.putText(calib_img, "Look at the red dot and press SPACE", 
                    (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        collected = 0
        required = 5
        
        while collected < required:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 표시
            cv2.imshow("Calibration", calib_img)
            cv2.imshow("Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # 스페이스바
                result_frame, gaze_vector = self.process_frame(frame)
                if gaze_vector is not None:
                    calibration_vectors.append(gaze_vector)
                    collected += 1
                    print(f"Collected {collected}/{required} calibration samples")
                    
                    # 보정점 이미지 업데이트
                    calib_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                    cv2.circle(calib_img, (300, 200), 10, (0, 0, 255), -1)
                    cv2.circle(calib_img, (300, 200), 30, (0, 0, 255), 2)
                    cv2.putText(calib_img, f"Collected {collected}/{required}", 
                                (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if calibration_vectors:
            self.calibration_vector = np.mean(calibration_vectors, axis=0)
            print(f"Calibration complete. Reference vector: {self.calibration_vector}")
            return True
        return False

    def apply_calibration(self, gaze_vector):
        """캘리브레이션을 적용한 시선 벡터 반환"""
        if hasattr(self, 'calibration_vector'):
            # 캘리브레이션 벡터를 기준으로 상대적 벡터 계산
            relative_vector = gaze_vector - self.calibration_vector
            # 크기 증폭 (더 민감하게)
            return relative_vector * 2.0
        return gaze_vector
    
    def evaluate_accuracy(self, actual_gaze_points, predicted_gaze_points):
        """시선 방향 정확도 평가"""
        # MAE 계산 (Mean Absolute Error)
        mae = np.mean(np.abs(actual_gaze_points - predicted_gaze_points), axis=0)
        print(f"시선 방향 예측의 평균 절대 오차 (MAE): X={mae[0]}, Y={mae[1]}")
        
        # MSE 계산 (Mean Squared Error)
        mse = np.mean((actual_gaze_points - predicted_gaze_points) ** 2, axis=0)
        print(f"시선 방향 예측의 평균 제곱 오차 (MSE): X={mse[0]}, Y={mse[1]}")
        
        # 정확도 계산 (정확도 기준: 20픽셀 이내)
        accuracy_threshold = 20  # 20픽셀 이내 오차는 정확하다고 판단
        correct_predictions = np.sum(np.linalg.norm(actual_gaze_points - predicted_gaze_points, axis=1) < accuracy_threshold)
        accuracy = correct_predictions / len(actual_gaze_points) * 100
        print(f"정확도: {accuracy}%")
