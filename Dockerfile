# Python 3.9 슬림 이미지
FROM python:3.9-slim

# 작업 디렉토리
WORKDIR /app

# 필수 시스템 패키지만 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisper tiny 모델 사전 다운로드
RUN python -c "import whisper; whisper.load_model('tiny')"

# 프로젝트 파일 복사
COPY . .

# 포트 설정
ENV PORT=5000
EXPOSE $PORT

# Flask 실행
CMD ["python", "app.py"]