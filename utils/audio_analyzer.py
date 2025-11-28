import os
import tempfile
import time
import uuid
import random
import torch # PyTorch 추가
import whisper # Whisper 라이브러리 사용
import librosa  # ✅ 추가
import numpy as np  # ✅ 추가

# 1단계에서 생성된 다운로드 유틸리티 import 시도
try:
    # model_downloader 모듈이 같은 utils 폴더에 있다고 가정합니다.
    from utils.model_downloader import download_model
    EXTERNAL_MODEL_DOWNLOAD = True
except ImportError as e:
    print(f"[ERROR] model_downloader 모듈 로드 실패: {e}")
    EXTERNAL_MODEL_DOWNLOAD = False

# Whisper 라이브러리가 로드에 실패했을 경우를 대비해 전역 변수 설정
WHISPER_LOADED = False
WHISPER_MODEL = None

class AudioAnalyzer:
    def __init__(self):
        global WHISPER_LOADED, WHISPER_MODEL
        
        self.use_dummy = True
        
        try:
            print("[INFO] Whisper 모델 로딩 중...")
            
            # ✅ 간단하게 공식 방법 사용
            WHISPER_MODEL = whisper.load_model("base")
            
            self.use_dummy = False
            WHISPER_LOADED = True
            print("[SUCCESS] Whisper 로드 성공")
            
        except Exception as e:
            print(f"[ERROR] Whisper 로드 실패: {e}")
            self.use_dummy = True
            
    def analyze(self, audio_file):
        """음성 분석 (librosa 추가)"""
        global WHISPER_MODEL
        print(f"[DEBUG] 음성 파일 수신: {audio_file.filename}, 크기: {audio_file.content_length}")
        
        if self.use_dummy or not WHISPER_LOADED:
            print("[INFO] 더미 모드로 분석")
            return self._get_realistic_dummy()
        
        # Whisper 로드가 성공했을 경우, 실제 분석 로직 실행 (기존과 동일)
        temp_path = None
        try:
            # 안전한 임시 파일 생성
            temp_dir = tempfile.gettempdir()
            safe_name = f"audio_{uuid.uuid4().hex[:8]}_{int(time.time())}.wav"
            temp_path = os.path.join(temp_dir, safe_name)
            
            # 파일 저장
            audio_file.save(temp_path)
            time.sleep(0.5) 
            
            # 파일 크기 확인 (중략)
            file_size = os.path.getsize(temp_path)
            
            if file_size < 2000:
                print("[WARN] 파일이 너무 작음")
                return self._get_short_audio_result()
            
            # ✅ librosa로 음성 특징 추출
            print("[DEBUG] librosa 음성 특징 추출 시작...")
            audio_features = self._extract_audio_features(temp_path)
            
            # Whisper 음성 인식 
            print("[DEBUG] Whisper 시작...")
            # transcribe 함수를 사용할 때 모델 객체를 명시적으로 전달
            result = whisper.transcribe( 
                WHISPER_MODEL, 
                temp_path,
                language='ko',
                task='transcribe',
                fp16=False,
                verbose=True,
            )
            
            text = result.get('text', '').strip()
            
            if not text or len(text) < 3:
                print("[WARN] 인식된 텍스트가 너무 짧음")
                return self._get_short_audio_result()
            
            # 분석 수행(audio_features 포함)
            analysis = self._analyze_korean_speech(text, result, audio_features)
            print(f"[SUCCESS] 실제 음성 분석 완료")
            return analysis
            
        except Exception as e:
            print(f"[ERROR] 음성 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(str(e))
            
        finally:
            # 안전한 파일 삭제
            if temp_path and os.path.exists(temp_path):
                try:
                    time.sleep(0.2)
                    os.remove(temp_path)
                    print(f"[DEBUG] 임시 파일 삭제됨")
                except Exception as del_error:
                    print(f"[WARN] 파일 삭제 실패: {del_error}")
    
    def _extract_audio_features(self, audio_path):
        """✅ librosa로 음성 특징 추출"""
        try:
            # 1. 음성 로드 (16kHz로 리샘플링)
            y, sr = librosa.load(audio_path, sr=16000)
            print(f"[DEBUG] 오디오 로드 완료: {len(y)} samples, {sr}Hz")
            
            # 2. 에너지(RMS) 분석
            energy = librosa.feature.rms(y=y)[0]
            
            # 3. 무음/발화 구간 분리 (20dB 기준)
            threshold = 0.02  # RMS 임계값
            silence_frames = np.sum(energy < threshold)
            speech_frames = np.sum(energy >= threshold)
            
            total_frames = len(energy)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            print(f"[DEBUG] 발화 비율: {speech_ratio:.2%}")
            
            # 4. 피치 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # 유효한 피치만 추출
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # 0이 아닌 피치만
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                pitch_variation = np.std(pitch_values)
            else:
                avg_pitch = 150.0
                pitch_variation = 0.0
            
            print(f"[DEBUG] 평균 피치: {avg_pitch:.1f}Hz")
            
            # 5. 음량 분석
            avg_volume = np.mean(energy)
            volume_variation = np.std(energy)
            
            print(f"[DEBUG] 평균 음량: {avg_volume:.3f}")
            
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
            # 실패 시 더미값 반환
            return {
                'speech_ratio': 0.7,
                'avg_pitch': 150.0,
                'pitch_variation': 25.0,
                'avg_volume': 0.1,
                'volume_variation': 0.05,
                'silence_frames': 0,
                'speech_frames': 0
            }
        
    def _get_error_result(self, error_msg):
        """실제 오류 결과"""
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
    
    def _analyze_korean_speech(self, text, whisper_result, audio_features):
        """한국어 음성 분석 (librosa 특징 포함)"""

        # 1. 기본 정보
        word_count = len([w for w in text.split() if w.strip()])
        duration = whisper_result.get('segments', [{}])
        total_duration = duration[-1].get('end', 5.0) if duration else 5.0
        
        speaking_rate = (word_count / total_duration * 60) if total_duration > 0 else 60
        
        # 발음 명확도 (한국어 문자 비율 + Whisper 품질)
        korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7a3'])
        total_chars = len(text.replace(' ', ''))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        base_clarity = 40 + (korean_ratio * 40)
        
        # Whisper 신뢰도 추가
        segments = whisper_result.get('segments', [])
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -1) for s in segments) / len(segments)
            whisper_bonus = max(0, (avg_logprob + 1) * 20)
            base_clarity += whisper_bonus
        
        pronunciation_clarity = max(50, min(95, base_clarity))

        # ✅ 3. 유창성 (librosa 기반으로 개선)
        speech_ratio = audio_features['speech_ratio']
        
        # 발화 비율 점수 (발화가 많을수록 좋음)
        speech_score =speech_ratio * 60  # 0~60점

        # 말하기 속도 점수
        if 80 <= speaking_rate <= 180:
            speed_score = 30  # 적정 속도
        else:
            speed_score = 15  # 너무 빠르거나 느림
        
        # 문장 완성도
        sentence_bonus = 0
        if '.' in text or '!' in text or '?' in text:
            sentence_bonus += 5
        if len(text) > 15:
            sentence_bonus += 5
        
        fluency = min(95, max(30, speech_score + speed_score + sentence_bonus))
        
        # 이해도 (내용 분석)
        comprehension = self._calculate_comprehension(text, pronunciation_clarity, fluency)
        
        return {
            'transcription': text,
            'duration': f"{total_duration:.1f}초",
            'word_count': word_count,
            'speaking_rate': f"{speaking_rate:.1f} 단어/분",
            'pronunciation_clarity': f"{pronunciation_clarity:.1f}%",
            'fluency': f"{fluency:.1f}%",
            'comprehension': f"{comprehension:.1f}%",
            'speech_features': {
                'avg_pitch': audio_features['avg_pitch'],
                'pitch_variation': audio_features['pitch_variation'],
                'avg_volume': audio_features['avg_volume'],
                'volume_variation': audio_features['volume_variation'],
                'speech_ratio': audio_features['speech_ratio']
            }
        }
    
    def _calculate_comprehension(self, text, clarity, fluency):
        """이해도 계산 - 내용 기반"""
        base_score = (clarity + fluency) / 2
        
        # 독서 관련 키워드 분석
        reading_keywords = [
            '독서', '책', '읽', '이야기', '내용', '생각', '느낌',
            '재미', '흥미', '배우', '알', '좋', '재밌', '신기',
            '등장인물', '주인공', '줄거리', '문장', '단어', '의미'
        ]
        
        keyword_count = sum(1 for keyword in reading_keywords if keyword in text)
        content_bonus = min(20, keyword_count * 3)
        
        # 문장 구조 분석
        structure_bonus = 0
        if len(text) > 20:
            structure_bonus += 5
        if any(punct in text for punct in '.!?'):
            structure_bonus += 5
        if '그래서' in text or '왜냐하면' in text or '하지만' in text:
            structure_bonus += 5
        
        comprehension = base_score + content_bonus + structure_bonus
        return max(30, min(95, comprehension))
    
    def _get_realistic_dummy(self):
        """현실적인 더미 결과"""
        dummy_responses = [
            "아기돼지삼형제가 집을 지었어요. 첫째는 짚으로 둘째는 나무로 지었어요. 셋째는 벽돌로 튼튼하게지었어요."
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
            'transcription': '녹음 시간이 너무 짧습니다.',
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
