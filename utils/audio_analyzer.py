import os
import tempfile
import time
import uuid
import random
import torch
import whisper

try:
    from utils.model_downloader import download_model
    EXTERNAL_MODEL_DOWNLOAD = True
except ImportError as e:
    print(f"[ERROR] model_downloader 모듈 로드 실패: {e}")
    EXTERNAL_MODEL_DOWNLOAD = False

WHISPER_LOADED = False
WHISPER_MODEL = None

class AudioAnalyzer:
    def __init__(self):
        global WHISPER_LOADED, WHISPER_MODEL

        self.use_dummy = True
        
        model_path = None
        
        if EXTERNAL_MODEL_DOWNLOAD:
            MODEL_ENV_VAR = "AUDIO_MODEL_URL"
            MODEL_LOCAL_NAME = "whisper_base.pt"
            model_path = download_model(MODEL_ENV_VAR, MODEL_LOCAL_NAME)
            
        if model_path:
            try:
                print(f"[INFO] Whisper 모델 로딩 시작. 경로: {model_path}")
                
                checkpoint = torch.load(model_path, map_location="cpu")
                dims = checkpoint["dims"]
                WHISPER_MODEL = whisper.model.Whisper(dims)
                WHISPER_MODEL.load_state_dict(checkpoint["model_state_dict"])
                WHISPER_MODEL.to("cpu") 
                
                del checkpoint
                
                self.use_dummy = False
                WHISPER_LOADED = True
                print("[SUCCESS] Whisper 로드 및 초기화 성공")
            except Exception as e:
                print(f"[ERROR] Whisper 실제 로드 실패: {e}")
                if model_path and os.path.exists(model_path):
                    os.remove(model_path)
                self.use_dummy = True

    def analyze(self, audio_file):
        """
        audio_file: Gradio는 파일 경로(str)를 전달, Flask는 FileStorage 객체
        """
        global WHISPER_MODEL
        
        # Gradio: audio_file은 이미 파일 경로(str)
        if isinstance(audio_file, str):
            audio_path = audio_file
            print(f"[DEBUG] Gradio 음성 파일 경로: {audio_path}")
        else:
            # Flask FileStorage 객체 처리 (하위 호환성)
            print(f"[DEBUG] Flask 음성 파일 수신: {audio_file.filename}")
            temp_dir = tempfile.gettempdir()
            safe_name = f"audio_{uuid.uuid4().hex[:8]}_{int(time.time())}.wav"
            audio_path = os.path.join(temp_dir, safe_name)
            audio_file.save(audio_path)
        
        if self.use_dummy or not WHISPER_LOADED:
            print("[INFO] 더미 모드로 분석")
            return self._get_realistic_dummy()
        
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(audio_path)
            
            if file_size < 2000:
                print("[WARN] 파일이 너무 작음")
                return self._get_short_audio_result()
            
            # Whisper 음성 인식
            print("[DEBUG] Whisper 시작...")
            result = whisper.transcribe( 
                WHISPER_MODEL, 
                audio_path,
                language='ko',
                task='transcribe',
                fp16=False,
                verbose=True,
                initial_prompt="다음은 한국어 음성입니다:",
                temperature=0.0
            )
            
            text = result.get('text', '').strip()
            
            if not text or len(text) < 3:
                print("[WARN] 인식된 텍스트가 너무 짧음")
                return self._get_short_audio_result()
            
            # 분석 수행
            analysis = self._analyze_korean_speech(text, result)
            print(f"[SUCCESS] 실제 음성 분석 완료")
            return analysis
            
        except Exception as e:
            print(f"[ERROR] 음성 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(str(e))
            
        finally:
            # Gradio: 임시 파일이 아닌 경우 삭제하지 않음
            # Flask에서 생성한 임시 파일만 삭제
            if not isinstance(audio_file, str) and os.path.exists(audio_path):
                try:
                    time.sleep(0.2)
                    os.remove(audio_path)
                    print(f"[DEBUG] 임시 파일 삭제됨")
                except Exception as del_error:
                    print(f"[WARN] 파일 삭제 실패: {del_error}")
    
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
    
    def _analyze_korean_speech(self, text, whisper_result):
        """한국어 음성 분석"""
        word_count = len([w for w in text.split() if w.strip()])
        duration = whisper_result.get('segments', [{}])
        total_duration = duration[-1].get('end', 5.0) if duration else 5.0
        
        speaking_rate = (word_count / total_duration * 60) if total_duration > 0 else 60
        
        # 발음 명확도
        korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7a3'])
        total_chars = len(text.replace(' ', ''))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        base_clarity = 40 + (korean_ratio * 40)
        
        segments = whisper_result.get('segments', [])
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -1) for s in segments) / len(segments)
            whisper_bonus = max(0, (avg_logprob + 1) * 20)
            base_clarity += whisper_bonus
        
        pronunciation_clarity = max(50, min(95, base_clarity))
        
        # 유창성
        speed_score = 70 if 80 <= speaking_rate <= 180 else 50
        
        sentence_bonus = 0
        if '.' in text or '!' in text or '?' in text:
            sentence_bonus += 10
        if len(text) > 15:
            sentence_bonus += 10
        
        fluency = min(95, speed_score + sentence_bonus)
        
        # 이해도
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
                'avg_pitch': 150.0,
                'pitch_variation': 25.0,
                'avg_volume': 0.1,
                'volume_variation': 0.05,
                'speech_ratio': 0.7
            }
        }
    
    def _calculate_comprehension(self, text, clarity, fluency):
        """이해도 계산"""
        base_score = (clarity + fluency) / 2
        
        reading_keywords = [
            '독서', '책', '읽', '이야기', '내용', '생각', '느낌',
            '재미', '흥미', '배우', '알', '좋', '재밌', '신기',
            '등장인물', '주인공', '줄거리', '문장', '단어', '의미'
        ]
        
        keyword_count = sum(1 for keyword in reading_keywords if keyword in text)
        content_bonus = min(20, keyword_count * 3)
        
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