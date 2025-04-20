import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GazeResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GazeResNet, self).__init__()
        
        # ResNet18 백본 로드
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 마지막 분류 레이어를 3D 시선 벡터 예측으로 변경
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        gaze = self.backbone(x)
        # 단위 벡터로 정규화
        return F.normalize(gaze, p=2, dim=1)