import cv2
import numpy as np
import torch
import time
import torchvision.transforms as transforms

class EyeProcessor:
    def __init__(self):
        # 모델 입력용 이미지 변환
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_eye(self, eye_img):
        if eye_img is None or eye_img.size == 0 or eye_img.shape[0] <= 0 or eye_img.shape[1] <= 0:
            return None
        
        try:
            # 이미지가 너무 작으면 크기 조정
            if eye_img.shape[0] < 20 or eye_img.shape[1] < 20:
                eye_img = cv2.resize(eye_img, (36, 36), interpolation=cv2.INTER_CUBIC)
            
            # 대비 향상
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 노이즈 감소
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 3채널로 변환
            eye_processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

            # 처리된 이미지 저장
            #cv2.imwrite(f"eye_processed_{time.time()}.jpg", eye_processed)
            
            # 모델 입력 형식으로 변환
            tensor = self.transform(eye_processed).unsqueeze(0)  # 배치 차원 추가
            return tensor
        except Exception as e:
            print(f"Eye preprocessing error: {e}")
            return None
        
    