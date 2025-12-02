# Python 3.9 슬림 이미지
FROM python:3.9-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 라이브러리 경로 환경변수
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir soundfile==0.12.1
RUN pip install --no-cache-dir -r requirements.txt

# ✅ soundfile 테스트만
RUN python -c "import soundfile; print('soundfile OK')"

# ✅ Google Speech 테스트 추가
RUN python -c "from google.cloud import speech; print('Google Speech OK')"

# 프로젝트 파일 복사
COPY . .

# 포트 설정
ENV PORT=5000
EXPOSE $PORT

# Flask 실행
CMD ["python", "app.py"]