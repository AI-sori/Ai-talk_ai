FROM python:3.9-slim

WORKDIR /app

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

# 라이브러리 경로 환경변수 추가
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

COPY requirements.txt .
RUN pip install --no-cache-dir soundfile==0.12.1
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import soundfile; print('soundfile OK')"
RUN python -c "import whisper; whisper.load_model('tiny')"

COPY . .

ENV PORT=5000
EXPOSE $PORT

CMD ["python", "app.py"]