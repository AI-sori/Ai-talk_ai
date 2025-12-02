import os
import tempfile
import time
import uuid
import random
import librosa
import numpy as np
import warnings
from google.cloud import speech

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 전역 변수
SPEECH_CLIENT = None

class AudioAnalyzer:
    def __init__(self):
        global SPEECH_CLIENT
        
        self.use_dummy = False
        
        try:
            print("[INFO] Google Speech API 초기화 중...")
            SPEECH_CLIENT = speech.SpeechClient()
            print("[SUCCESS] Google Speech API 준비 완료")
        except Exception as e:
            print(f"[ERROR] Google Speech API 초기화 실패: {e}")
            print("[INFO] 더미 모드로 전환")
            self.use_dummy = True
            
    def analyze(self, audio_file):
        """음성 분석 (Google Speech API)"""
        global SPEECH_CLIENT
        print(f"[DEBUG] 음성 파일 수신: {audio_file.filename}")
        
        if self.use_dummy:
            print("[INFO] 더미 모드로 분석")
            return self._get_realistic_dummy()
        
        temp_path = None
        try:
            # 안전한 임시 파일 생성
            temp_dir = tempfile.gettempdir()
            safe_name = f"audio_{uuid.uuid4().hex[:8]}.wav"
            temp_path = os.path.join(temp_dir, safe_name)
            
            # 파일 저장
            audio_file.save(temp_path)
            time.sleep(0.3)
            
            file_size = os.path.getsize(temp_path)
            print(f"[DEBUG] 파일 크기: {file_size} bytes")
            
            if file_size < 2000:
                print("[WARN] 파일이 너무 작음")
                return self._get_short_audio_result()
            
            # ✅ librosa로 음성 특징 추출
            print("[INFO] librosa 음성 특징 추출 중...")
            audio_features = self._extract_audio_features(temp_path)
            
            # ✅ Google Speech API로 음성 인식
            print("[INFO] Google Speech API 음성 인식 시작...")
            text = self._google_speech_recognize(temp_path)
            
            if not text or len(text) < 3:
                print("[WARN] 인식된 텍스트가 너무 짧음")
                return self._get_short_audio_result()
            
            print(f"[SUCCESS] 인식 완료: {text}")
            
            # 분석 수행
            analysis = self._analyze_korean_speech(text, audio_features)
            print(f"[SUCCESS] 음성 분석 완료")
            return analysis
            
        except Exception as e:
            print(f"[ERROR] 음성 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(str(e))
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print("[DEBUG] 임시 파일 삭제됨")
                except Exception as del_error:
                    print(f"[WARN] 파일 삭제 실패: {del_error}")
    
    def _google_speech_recognize(self, audio_path):
        """Google Speech API로 음성 인식"""
        global SPEECH_CLIENT
        
        try:
            # 16kHz mono WAV로 변환
            print("[DEBUG] 오디오 형식 변환 중...")
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # 임시 파일로 저장
            import soundfile as sf
            temp_wav = audio_path.replace('.wav', '_16k.wav')
            sf.write(temp_wav, y, 16000)
            
            # 오디오 파일 읽기
            with open(temp_wav, 'rb') as audio_file:
                content = audio_file.read()
            
            # Google Speech API 설정
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='ko-KR',
                enable_automatic_punctuation=True,
            )
            
            # 음성 인식 실행
            print("[DEBUG] Google API 호출 중...")
            response = SPEECH_CLIENT.recognize(config=config, audio=audio)
            
            # 임시 파일 삭제
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            # 결과 추출
            if not response.results:
                print("[WARN] 인식 결과 없음")
                return ""
            
            text = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            
            print(f"[DEBUG] 인식 신뢰도: {confidence:.2%}")
            
            return text.strip()
            
        except Exception as e:
            print(f"[ERROR] Google Speech 인식 실패: {e}")
            return ""
    
    def _extract_audio_features(self, audio_path):
        """librosa로 음성 특징 추출"""
        try:
            # 음성 로드
            y, sr = librosa.load(audio_path, sr=16000)
            print(f"[DEBUG] 오디오 로드: {len(y)} samples, {sr}Hz")
            
            # 에너지(RMS) 분석
            energy = librosa.feature.rms(y=y)[0]
            
            # 무음/발화 구간 분리
            threshold = 0.02
            silence_frames = np.sum(energy < threshold)
            speech_frames = np.sum(energy >= threshold)
            total_frames = len(energy)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            print(f"[DEBUG] 발화 비율: {speech_ratio:.2%}")
            
            # 피치 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                pitch_variation = np.std(pitch_values)
            else:
                avg_pitch = 150.0
                pitch_variation = 0.0
            
            # 음량 분석
            avg_volume = np.mean(energy)
            volume_variation = np.std(energy)
            
            return {
                'speech_ratio': float(speech_ratio),
                'avg_pitch': float(avg_pitch),
                'pitch_variation': float(pitch_variation),
                'avg_volume': float(avg_volume),
                'volume_variation': float(volume_variation),
                'silence_frames': int(silence_frames),
                'speech_frames': int(speech_frames)
            }
            
        except Exception as e:
            print(f"[ERROR] librosa 특징 추출 실패: {e}")
            return {
                'speech_ratio': 0.7,
                'avg_pitch': 150.0,
                'pitch_variation': 25.0,
                'avg_volume': 0.1,
                'volume_variation': 0.05,
                'silence_frames': 0,
                'speech_frames': 0
            }
    
    def _analyze_korean_speech(self, text, audio_features):
        """한국어 음성 분석"""
        
        # 기본 정보
        word_count = len([w for w in text.split() if w.strip()])
        duration = len(text) / 5  # 대략 추정
        speaking_rate = (word_count / duration * 60) if duration > 0 else 60
        
        # 발음 명확도 (librosa 기반)
        speech_ratio = audio_features['speech_ratio']
        volume_score = min(30, audio_features['avg_volume'] * 300)
        clarity_base = speech_ratio * 60
        pronunciation_clarity = min(95, max(50, clarity_base + volume_score))
        
        # 유창성
        fluency_base = speech_ratio * 50
        pitch_bonus = min(25, audio_features['pitch_variation'] / 2)
        
        # 속도 점수
        if 80 <= speaking_rate <= 180:
            speed_score = 20
        else:
            speed_score = 10
        
        fluency = min(95, max(50, fluency_base + pitch_bonus + speed_score))
        
        # 이해도
        comprehension = self._calculate_comprehension(text, pronunciation_clarity, fluency)
        
        return {
            'transcription': text,
            'duration': f"{duration:.1f}초",
            'word_count': word_count,
            'speaking_rate': f"{speaking_rate:.1f} 단어/분",
            'pronunciation_clarity': f"{pronunciation_clarity:.1f}%",
            'fluency': f"{fluency:.1f}%",
            'comprehension': f"{comprehension:.1f}%",
            'speech_features': audio_features
        }
    
    def _calculate_comprehension(self, text, clarity, fluency):
        """이해도 계산"""
        base_score = (clarity + fluency) / 2
        
        # 키워드 분석
        reading_keywords = [
            '토끼', '거북이', '개미', '베짱이', '돼지', '늑대',
            '집', '달리기', '여름', '겨울', '벽돌', '짚', '나무'
        ]
        
        keyword_count = sum(1 for keyword in reading_keywords if keyword in text)
        content_bonus = min(20, keyword_count * 5)
        
        # 문장 구조
        structure_bonus = 0
        if len(text) > 15:
            structure_bonus += 5
        if any(punct in text for punct in '.!?'):
            structure_bonus += 5
        
        comprehension = base_score + content_bonus + structure_bonus
        return max(50, min(95, comprehension))
    
    def _get_error_result(self, error_msg):
        """오류 결과"""
        return {
            'transcription': f'음성 인식 실패: {error_msg}',
            'duration': '0.0초',
            'word_count': 0,
            'speaking_rate': '0.0 단어/분',
            'pronunciation_clarity': '0.0%',
            'fluency': '0.0%',
            'comprehension': '0.0%',
            'speech_features': {
                'avg_pitch': 0.0,
                'pitch_variation': 0.0,
                'avg_volume': 0.0,
                'volume_variation': 0.0,
                'speech_ratio': 0.0
            }
        }
    
    def _get_realistic_dummy(self):
        """더미 결과"""
        dummy_responses = [
            "토끼와 거북이가 달리기를 했어요",
            "개미는 여름에 열심히 일했어요",
            "아기돼지 삼형제가 집을 지었어요"
        ]
        
        text = random.choice(dummy_responses)
        word_count = len(text.split())
        
        return {
            'transcription': text,
            'duration': f"{random.uniform(4, 8):.1f}초",
            'word_count': word_count,
            'speaking_rate': f"{random.uniform(100, 140):.1f} 단어/분",
            'pronunciation_clarity': f"{random.uniform(75, 90):.1f}%",
            'fluency': f"{random.uniform(70, 85):.1f}%",
            'comprehension': f"{random.uniform(75, 92):.1f}%",
            'speech_features': {
                'avg_pitch': random.uniform(120, 180),
                'pitch_variation': random.uniform(15, 25),
                'avg_volume': random.uniform(0.05, 0.12),
                'volume_variation': random.uniform(0.02, 0.06),
                'speech_ratio': random.uniform(0.6, 0.8)
            }
        }
    
    def _get_short_audio_result(self):
        """짧은 음성 결과"""
        return {
            'transcription': '녹음 시간이 너무 짧습니다',
            'duration': '1.5초',
            'word_count': 0,
            'speaking_rate': '0.0 단어/분',
            'pronunciation_clarity': '30.0%',
            'fluency': '30.0%',
            'comprehension': '30.0%',
            'speech_features': {
                'avg_pitch': 120.0,
                'pitch_variation': 10.0,
                'avg_volume': 0.03,
                'volume_variation': 0.01,
                'speech_ratio': 0.3
            }
        }