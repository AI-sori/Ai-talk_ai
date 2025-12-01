FROM python:3.9-slim

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

# requirements.txt 복사
COPY requirements.txt .

# soundfile 먼저 설치 (libsndfile1-dev 설치 후)
RUN pip install --no-cache-dir soundfile==0.12.1

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# soundfile 제대로 설치됐는지 테스트
RUN python -c "import soundfile; print('soundfile OK')"

# Whisper 다운로드
RUN python -c "import whisper; whisper.load_model('tiny')"

COPY . .

ENV PORT=5000
EXPOSE $PORT

CMD ["python", "app.py"]