from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

def put_korean_text(img, text, position, font_path="malgun.ttf", font_size=30, color=(0, 255, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(img)
    
    # 그리기 객체 생성
    draw = ImageDraw.Draw(img_pil)
    
    # 시도할 폰트 경로 목록
    font_paths = [
        font_path,  # 사용자 지정 폰트
        "malgun.ttf",
        "C:/Windows/Fonts/malgun.ttf",  # Windows 경로
        "C:/Windows/Fonts/gulim.ttc", 
        "C:/Windows/Fonts/batang.ttc",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux 경로
        "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS 경로
    ]
    
    # 폰트 로드 시도
    font = None
    success_path = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            success_path = path
            print(f"성공적으로 로드된 폰트: {path}")
            break
        except Exception as e:
            # 조용히 실패 처리
            continue
    
    # 모든 폰트 로드 실패 시 기본 폰트 사용
    if font is None:
        print("모든 폰트 로드 실패, 기본 폰트 사용")
        font = ImageFont.load_default()
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)
    
    # PIL 이미지를 OpenCV 이미지로 다시 변환
    return np.array(img_pil)