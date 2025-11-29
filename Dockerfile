# Python 3.9 슬림 이미지
FROM python:3.9-slim
# 작업 디렉토리
WORKDIR /app
# 시스템 패키지 설치 (OpenCV, librosa용)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 전체 프로젝트 복사 (templates, static 포함)
COPY . .
# 포트 설정
ENV PORT=5000
EXPOSE $PORT

# Flask 실행
CMD ["python", "app.py"]