import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class GazeResNet(nn.Module):
    def __init__(self, num_classes=2):  # x, y 좌표
        super(GazeResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        
        # 입력 채널을 1로 변경 (그레이스케일 눈 이미지)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 출력 레이어를 시선 좌표로 변경
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class GazeModel:
    def __init__(self, model_path='models/best_resnet_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GazeResNet()
        
        try:
            # 모델 로드 (PyTorch 2.6 호환성 수정)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # 모델 파일이 없으면 랜덤 초기화된 모델 사용
            self.model.to(self.device)
            self.model.eval()
            print("Using randomly initialized model")
    
    def preprocess_eye_image(self, eye_image):
        """눈 이미지 전처리"""
        try:
            # 그레이스케일 변환
            if len(eye_image.shape) == 3:
                eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            
            # 크기 조정 (60x36 - MPIIGaze 표준)
            eye_image = cv2.resize(eye_image, (60, 36))
            
            # 정규화
            eye_image = eye_image.astype(np.float32) / 255.0
            
            # 텐서 변환
            eye_tensor = torch.from_numpy(eye_image).unsqueeze(0).unsqueeze(0)  # (1, 1, 36, 60)
            eye_tensor = eye_tensor.to(self.device)
            
            return eye_tensor
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def predict_gaze(self, left_eye, right_eye):
        """시선 방향 예측"""
        try:
            print(f"[DEBUG] 눈 이미지 입력 - 왼쪽: {left_eye.shape}, 오른쪽: {right_eye.shape}")
            
            with torch.no_grad():
                # 왼쪽 눈 처리
                left_tensor = self.preprocess_eye_image(left_eye)
                if left_tensor is None:
                    print("[DEBUG] 왼쪽 눈 전처리 실패")
                    return None
                print(f"[DEBUG] 왼쪽 눈 텐서 크기: {left_tensor.shape}")
                
                # 오른쪽 눈 처리
                right_tensor = self.preprocess_eye_image(right_eye)
                if right_tensor is None:
                    print("[DEBUG] 오른쪽 눈 전처리 실패")
                    return None
                print(f"[DEBUG] 오른쪽 눈 텐서 크기: {right_tensor.shape}")
                
                # 예측
                left_pred = self.model(left_tensor)
                right_pred = self.model(right_tensor)
                print(f"[DEBUG] 모델 예측 - 왼쪽: {left_pred}, 오른쪽: {right_pred}")
                
                # 평균 계산
                avg_pred = (left_pred + right_pred) / 2
                gaze_x, gaze_y = avg_pred[0].cpu().numpy()
                
                print(f"[DEBUG] 최종 시선 좌표: ({gaze_x}, {gaze_y})")
                return (float(gaze_x), float(gaze_y))
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None