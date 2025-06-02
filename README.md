# 📚 영유아 독서 분석 시스템

실시간 시선 추적과 음성 분석을 통한 영유아 독서 능력 진단 시스템

## ✨ 주요 기능

- 🎯 **실시간 시선 추적**: 웹캠을 통한 읽기 패턴 분석
- 🎤 **음성 분석**: 발음, 유창성, 이해도 측정
- 📖 **3개 이야기**: 토끼와 거북이, 개미와 베짱이, 아기돼지 삼형제
- 📊 **종합 리포트**: 맞춤형 학습 추천 제공
- 📱 **사용자 친화적**: 부모님이 쉽게 이해할 수 있는 리포트

## 🛠️ 설치 및 실행

### 필수 요구사항
- Python 3.8+
- 웹캠
- 마이크

## 가상환경 생성 (권장)
python -m venv .venv
### 가상환경 활성화
### Windows:
.venv\Scripts\activate
### macOS/Linux:
source venv/bin/activate

### 실제 음성 인식을 원하는 경우
pip install openai-whisper librosa
### 실제 얼굴 감지를 원하는 경우  
pip install mediapipe
## 모든 기능 한번에 설치
pip install -r requirements.txt

### 브라우저에서 `http://localhost:5000` 접속

## 📖 사용 방법

1. **🚀 시스템 초기화**: 카메라/마이크 권한 허용
2. **🎯 보정하기**: 화면에 나타나는 5개 포인트 응시
3. **👁️ 추적 시작**: 3개 이야기를 자연스럽게 읽기
4. **🎤 음성 녹음**: 읽은 내용에 대해 자유롭게 말하기
5. **📋 리포트 생성**: 아동 정보 입력 후 결과 확인

## 🎯 측정 항목

### 시선 추적 분석
- **읽기 속도**: 시선 이동 패턴 기반
- **집중력**: 중앙 응시 비율
- **시선 패턴**: 좌우 편향, 불안정성 감지

### 음성 분석  
- **발음 명확도**: Whisper 기반 음성 인식
- **말하기 유창성**: 속도, 일시정지 분석
- **이해력**: 내용 키워드 분석

## 📊 API 응답 형식

```json
{
  "id": 1748892575,
  "user_id": 1,
  "diagnosis_type": "reading_analysis",
  "created_at": "2025-06-03T04:29:35.111121",
  "report": {
    "child_name": "홍길동",
    "diagnosis_date": "2025-06-03",
    "reading_time": "65.5초",
    "results": {
      "reading_speed": "45.2 회/분",
      "concentration": "78.3%",
      "comprehension": "86.8%"
    },
    "eye_tracking": {
      "issues": "정상",
      "focus_time": "52.1초"
    },
    "speech_analysis": {
      "transcription": "토끼와 거북이 이야기가...",
      "fluency": "70.0%",
      "pronunciation_clarity": "83.6%",
      "speaking_rate": "58.8 단어/분",
      "duration": "8.2초",
      "word_count": 8
    },
    "feedback": {
      "summary": "총 131회 측정, 집중도 78.3%",
      "recommended_activities": [
        "15분 단위 집중 독서 연습",
        "시각적 집중력 향상 게임"
      ],
      "next_diagnosis_date": "2025-07-03"
    }
  }
}
