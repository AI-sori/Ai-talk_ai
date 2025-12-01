FROM python:3.9-slim
WORKDIR /app

# 시스템 패키지 먼저 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사
COPY requirements.txt .

# soundfile 명시적 재설치
RUN pip install --no-cache-dir --force-reinstall soundfile
RUN pip install --no-cache-dir -r requirements.txt

# Whisper 다운로드
RUN python -c "import whisper; whisper.load_model('tiny')"

COPY . .
ENV PORT=5000
EXPOSE $PORT
CMD ["python", "app.py"]