# =============================================================================
# Flask 서버 - AI 읽기 진단 백엔드
# =============================================================================

from flask import Flask, render_template, request, jsonify
import cv2
import sys
import base64
import numpy as np
import pymysql
import gc
from datetime import datetime, timedelta
import os
import json

# ✅ 로그 즉시 출력 설정
sys.stdout.flush()

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)


# Google Cloud 인증 설정
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    # Railway 환경변수에서 JSON 읽기
    creds_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
    
    # 임시 파일로 저장
    with open('/tmp/google-credentials.json', 'w') as f:
        f.write(creds_json)
    
    # 환경변수 설정
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/google-credentials.json'
    print("[INFO] Google Cloud 인증 설정 완료")

# =============================================================================
# 1. 모듈 임포트 및 초기화
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# AI 모듈 로드 (시선 추적, 음성 분석)
try:
    from utils.gaze_tracker import GazeTracker
    from utils.audio_analyzer import AudioAnalyzer
    print("[INFO] 모듈 로드 성공")
except ImportError as e:
    print(f"[ERROR] 모듈 로드 실패: {e}")
    # 더미 클래스 (테스트용)
    class GazeTracker:
        def __init__(self): 
            self.calibrated = False
        def get_gaze_direction(self, frame): 
            return {'gaze_x': 0.1, 'gaze_y': 0.1, 'face_center': (320, 240)}
        def calibrate(self, points): 
            self.calibrated = True
            return True
        def track_reading(self, frame):
            import random
            return {
                'direction': random.choice(['left', 'center', 'right']),
                'confidence': random.uniform(0.5, 0.9),
                'position': (random.randint(200, 800), random.randint(200, 600)),
                'error_offset': random.uniform(10, 50)
            }
    
    class AudioAnalyzer:
        def analyze(self, audio_file):
            return {
                'transcription': '테스트 음성 인식 결과',
                'duration': '5.0초',
                'word_count': 10,
                'speaking_rate': '120.0 단어/분',
                'pronunciation_clarity': '85.0%',
                'fluency': '78.0%',
                'comprehension': '81.5%'
            }

# =============================================================================
# 2. DB 연결 설정
# =============================================================================

DB_CONFIG = {
    'host': 'svc.sel4.cloudtype.app',
    'port': 30213,
    'user': 'root',
    'password': 'ai-talk',
    'database': 'ai-talk',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """MySQL DB 연결"""
    try:
        connection = pymysql.connect(**DB_CONFIG, connect_timeout=10)
        print("[SUCCESS] DB 연결 성공")
        return connection
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        raise Exception(f"DB연결실패: {e}")

# =============================================================================
# 3. Flask 앱 설정
# =============================================================================

app = Flask(__name__, 
           template_folder=os.path.join(current_dir, 'templates'),
           static_folder=os.path.join(current_dir, 'static'))

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 전역 변수
gaze_tracker = None
audio_analyzer = None
calibration_data = []  # 보정 데이터
tracking_results = []  # 추적 결과

# =============================================================================
# 4. 유틸리티 함수
# =============================================================================

def cleanup_memory():
    """메모리 정리 (최근 데이터만 유지)"""
    global calibration_data, tracking_results
    if len(calibration_data) > 100:
        calibration_data = calibration_data[-100:]
    if len(tracking_results) > 1000:
        tracking_results = tracking_results[-1000:]
    gc.collect()

def decode_frame(frame_data):
    """Base64 프레임 디코딩"""
    try:
        header, b64_data = frame_data.split(',', 1)
        frame_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[ERROR] 프레임 디코딩 실패: {e}")
        return None
    
    # 레벨 계산 함수
def calculate_level_and_issues(concentration, clarity, fluency):
    """
    3개 지표로 레벨 분류 및 약점 분석
    
    Args:
        concentration: 시선 집중도 (%)
        clarity: 발음 명확도 (%)
        fluency: 유창성 (%)
    
    Returns:
        {
            'level': 'beginner/intermediate/advanced',
            'total_score': 76.4,
            'issues': '집중력 부족 (68.5%)',
            'weak_area': '집중력'
        }
    """
    # 1. 종합 점수 (가중 평균)
    total_score = (
        concentration * 0.33 +    # 집중도 33%
        clarity * 0.33 +          # 명확성 33%
        fluency * 0.34            # 유창성 34%
    )
    
    # 2. 레벨 분류
    if total_score >= 80:
        level = "advanced"
    elif total_score >= 60:
        level = "intermediate"
    else:
        level = "beginner"
    
    # 3. 가장 낮은 지표 찾기
    scores = {
        '집중력': concentration,
        '명확성': clarity,
        '유창성': fluency
    }
    
    lowest_key = min(scores, key=scores.get)
    lowest_value = scores[lowest_key]
    
    # issues 텍스트 생성
    issues = f"{lowest_key} 부족 ({lowest_value:.1f}%)"
    
    return {
        'level': level,
        'total_score': round(total_score, 1),
        'issues': issues,
        'weak_area': lowest_key,
        'weak_score': lowest_value
    }

# =============================================================================
# 5. API 엔드포인트 - 초기화
# =============================================================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/init_tracker', methods=['POST'])
def init_tracker():
    """시선 추적 & 음성 분석 시스템 초기화"""
    global gaze_tracker, audio_analyzer
    try:
        print("[INFO] 트래커 초기화 시작")
        gaze_tracker = GazeTracker()
        audio_analyzer = AudioAnalyzer()
        cleanup_memory()
        print("[INFO] 초기화 완료")
        return jsonify({"status": "success", "message": "초기화 완료"})
    except Exception as e:
        print(f"[ERROR] 초기화 실패: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# =============================================================================
# 6. API 엔드포인트 - 시선 추적
# =============================================================================

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """시선 추적 보정"""
    global calibration_data
    try:
        data = request.json
        frame = decode_frame(data['frame'])
        
        if frame is None:
            return jsonify({"status": "error", "message": "프레임 처리 실패"})
        
        if gaze_tracker:
            gaze_point = gaze_tracker.get_gaze_direction(frame)
            
            if gaze_point:
                calibration_data.append({
                    'target': (data['target_x'], data['target_y']),
                    'gaze': gaze_point
                })
                print(f"[INFO] 보정 포인트 {len(calibration_data)}개")
                cleanup_memory()
                return jsonify({"status": "success", "calibration_points": len(calibration_data)})
            else:
                return jsonify({"status": "error", "message": "시선 감지 실패"})
        
        return jsonify({"status": "error", "message": "트래커 미초기화"})
        
    except Exception as e:
        print(f"[ERROR] 보정 실패: {e}")
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if 'frame' in locals():
            del frame
        gc.collect()

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    """시선 추적 시작"""
    global tracking_results
    try:
        tracking_results = []
        
        if len(calibration_data) >= 4:
            success = gaze_tracker.calibrate(calibration_data)
            if success:
                print("[INFO] 추적 시작")
                return jsonify({"status": "success", "message": "추적 시작"})
            else:
                return jsonify({"status": "error", "message": "보정 실패"})
        else:
            return jsonify({"status": "error", "message": f"최소 4개 필요 (현재: {len(calibration_data)}개)"})
            
    except Exception as e:
        print(f"[ERROR] 추적 시작 실패: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    """시선 추적 중지"""
    cleanup_memory()
    print("[INFO] 추적 중지")
    return jsonify({"status": "success", "message": "추적 중지"})

@app.route('/track_gaze', methods=['POST'])
def track_gaze():
    """실시간 시선 추적"""
    global tracking_results
    try:
        data = request.json
        frame = decode_frame(data['frame'])
        
        if frame is None:
            return jsonify({"status": "success", "direction": "center", "confidence": 0.3, "error_offset": 50})
        
        if gaze_tracker:
            result = gaze_tracker.track_reading(frame)
            
            if result:
                # 결과 저장
                tracking_results.append({
                    'timestamp': datetime.now().isoformat(),
                    'gaze_direction': result['direction'],
                    'confidence': result['confidence'],
                    'position': result['position']
                })
                
                return jsonify({
                    "status": "success",
                    "direction": result['direction'],
                    "confidence": float(result['confidence']),
                    "error_offset": float(result.get('error_offset', 0)),
                    "position": result['position']
                })
        
        return jsonify({"status": "error", "message": "트래커 미초기화"})
        
    except Exception as e:
        print(f"[ERROR] 추적 실패: {e}")
        return jsonify({"status": "success", "direction": "center", "confidence": 0.3, "error_offset": 50})
    finally:
        if 'frame' in locals():
            del frame
        if len(tracking_results) % 50 == 0:
            cleanup_memory()

# =============================================================================
# 7. API 엔드포인트 - 음성 분석
# =============================================================================

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """음성 분석 (Whisper + librosa)"""
    try:
        if 'audio' not in request.files:
            return jsonify({"status": "error", "message": "오디오 파일 없음"})
        
        audio_file = request.files['audio']
        print(f"[INFO] 음성 분석 시작: {audio_file.filename}")
        
        if audio_analyzer:
            result = audio_analyzer.analyze(audio_file)
            print("[INFO] 음성 분석 완료")
            return jsonify({"status": "success", "result": result})
        
        return jsonify({"status": "error", "message": "음성 분석기 미초기화"})
        
    except Exception as e:
        print(f"[ERROR] 음성 분석 실패: {e}")
        return jsonify({"status": "error", "message": str(e)})
    finally:
        gc.collect()

# =============================================================================
# 8. API 엔드포인트 - 리포트 생성
# =============================================================================

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """종합 리포트 생성 (시선 + 음성 분석 결과)"""
    global tracking_results
    try:
        data = request.json
        user_email = data.get('user_email', 'unknown@example.com')
        child_name = data.get('child_name', 'Unknown')
        # user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        print(f"[INFO] 리포트 생성:  이메일={user_email}, 이름={child_name}, {len(tracking_results)}개 추적 결과")
        
        # 시선 추적 분석
        if tracking_results:
            total_time = len(tracking_results) * 0.5
            left_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'left')
            center_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'center')
            right_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'right')
            
            concentration_score = (center_count / len(tracking_results) * 100)
            direction_changes = sum(1 for i in range(1, len(tracking_results)) 
                                  if tracking_results[i]['gaze_direction'] != tracking_results[i-1]['gaze_direction'])
            reading_speed = direction_changes / (total_time / 60) if total_time > 0 else 0
        else:
            total_time = 0
            concentration_score = 0
            reading_speed = 0
            left_count = center_count = right_count = 0

        # 음성 분석 결과에서 수치 추출
        clarity = float(audio_result.get('pronunciation_clarity', '0').replace('%', ''))
        fluency_score = float(audio_result.get('fluency', '0').replace('%', ''))
        
        # 레벨 계산
        level_info = calculate_level_and_issues(
            concentration_score,
            clarity,
            fluency_score
        )
        print(f"[INFO] 레벨 계산 완료: {level_info['level']} ({level_info['total_score']}점)")

        # 이슈 분석
        issues = []
        if concentration_score < 40:
            issues.append("심각한 집중력 부족")
        elif concentration_score < 60:
            issues.append("집중력 개선 필요")
        
        if reading_speed < 20:
            issues.append("독서 속도 느림")
            
        issues_text = ", ".join(issues) if issues else "정상"
        
        # 권장 활동
        recommended_activities = []
        if concentration_score < 60:
            recommended_activities.extend([
                "15분 단위 집중 독서 연습",
                "시각적 집중력 향상 게임"
            ])
        
        if not recommended_activities:
            recommended_activities = ["현재 수준 유지", "정기적인 독서 습관"]
        
        # 리포트 구성
        report = {
            "id": int(datetime.now().timestamp()),
            # "user_id": user_id,
            "diagnosis_type": "reading_analysis",
            "created_at": datetime.now().isoformat(),
            # 유저 프로필 추가
            "user_profile": {
                "email": user_email,
                "name": child_name
            },
            "report": {
                "child_name": child_name,
                "diagnosis_date": datetime.now().strftime("%Y-%m-%d"),
                "reading_time": f"{total_time:.1f}초",
                "results": {
                    "reading_speed": f"{reading_speed:.1f} 회/분",
                    "concentration": f"{concentration_score:.1f}%",
                    "comprehension": audio_result.get('comprehension', '0.0%')
                },
                "eye_tracking": {
                    "issues": issues_text,
                    "focus_time": f"{center_count * 0.5:.1f}초"
                },
                "speech_analysis": {
                    "transcription": audio_result.get('transcription', 'N/A'),
                    "fluency": audio_result.get('fluency', '0.0%'),
                    "pronunciation_clarity": audio_result.get('pronunciation_clarity', '0.0%'),
                    "speaking_rate": audio_result.get('speaking_rate', '0.0 단어/분'),
                    "duration": audio_result.get('duration', '0.0초'),
                    "word_count": audio_result.get('word_count', 0)
                },
                # 레벨 평가 섹션 추가
                "level_assessment": {
                    "level": level_info['level'],
                    "total_score": level_info['total_score'],
                    "concentration": concentration_score,
                    "clarity": clarity,
                    "fluency": fluency_score,
                    "issues": level_info['issues'],
                    "weak_area": level_info['weak_area']
                },
                "feedback": {
                    "summary": f"총 {len(tracking_results)}회 측정, 집중도 {concentration_score:.1f}%",
                    "recommended_activities": recommended_activities,
                    "next_diagnosis_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                }
            }
        }
        
        print("[INFO] 리포트 생성 완료")

        # ✅ 추가: Spring 서버로 전송 (핵심데이터만)
        try:
            import requests
            # 스프링 서버 주소 (배포된 AWS IP)
            spring_url = "http://15.165.102.27:8080/api/ai/receive"

            # 학습 프로그램 맞춤용 핵심 데이터
            learning_data = {
                "user_profile": {
                    "email": user_email,
                    "name": child_name
                },
                "diagnosis_info": {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "reading_time_seconds": round(total_time, 1)
                },
                "level_assessment": {
                    "level": level_info['level'],
                    "total_score": level_info['total_score'],
                    "weak_area": level_info['weak_area'],
                    "issues": level_info['issues'],
                    "scores": {
                        "concentration": round(concentration_score, 1),
                        "clarity": round(clarity, 1),
                        "fluency": round(fluency_score, 1)
                    }
                }
            }
            
            print(f"[INFO] Spring 서버로 전송 시도: {spring_url}")
            print(f"[DEBUG] 전송 데이터: {learning_data}")
            
            response = requests.post(
                spring_url,
                json=learning_data,  # 핵심 데이터만 전송
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                print("[SUCCESS] Spring 서버로 전송 성공!")
            else:
                print(f"[WARN] Spring 응답 코드: {response.status_code}")
                print(f"[WARN] Spring 응답 내용: {response.text}")
                
        except requests.exceptions.Timeout:
            print("[ERROR] Spring 서버 응답 시간 초과")
        except requests.exceptions.ConnectionError:
            print("[ERROR] Spring 서버 연결 실패 (서버가 꺼져있을 수 있음)")
        except Exception as spring_error:
            print(f"[ERROR] Spring 전송 실패: {spring_error}")
            # Spring 전송 실패해도 리포트는 반환

        return jsonify({"status": "success", "report": report})
        
    except Exception as e:
        print(f"[ERROR] 리포트 생성 실패: {e}")
        return jsonify({"status": "error", "message": str(e)})
    finally:
        cleanup_memory()

# =============================================================================
# 9. PDF 생성 관련 
# =============================================================================

@app.route('/download_pdf_report', methods=['POST'])
def download_pdf_report():
    """PDF 리포트 생성 (완전판 v2)"""
    global tracking_results
    
    try:
        data = request.json
        child_name = data.get('child_name', 'Unknown')
        user_email = data.get('user_email', 'unknown@example.com')
        audio_result = data.get('audio_result', {})
        
        print(f"[INFO] PDF 생성 시작: {child_name}")
        
        # 데이터 수집
        if tracking_results:
            total_time = len(tracking_results) * 0.5
            left_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'left')
            center_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'center')
            right_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'right')
            concentration_score = (center_count / len(tracking_results) * 100)
            direction_changes = sum(1 for i in range(1, len(tracking_results)) 
                                  if tracking_results[i]['gaze_direction'] != tracking_results[i-1]['gaze_direction'])
            reading_speed = direction_changes / (total_time / 60) if total_time > 0 else 0
            focus_time = center_count * 0.5
        else:
            total_time = focus_time = concentration_score = reading_speed = 0
            left_count = center_count = right_count = 0
        
        clarity = float(audio_result.get('pronunciation_clarity', '0').replace('%', ''))
        fluency_score = float(audio_result.get('fluency', '0').replace('%', ''))
        comprehension = float(audio_result.get('comprehension', '0').replace('%', ''))
        transcription = audio_result.get('transcription', 'N/A')
        speaking_rate = audio_result.get('speaking_rate', '0 단어/분')
        duration = audio_result.get('duration', '0초')
        word_count = audio_result.get('word_count', 0)
        
        level_info = calculate_level_and_issues(concentration_score, clarity, fluency_score)
        
        # 시선 패턴 분석
        issues = []
        if concentration_score < 40:
            issues.append("심각한 집중력 부족")
        elif concentration_score < 60:
            issues.append("집중력 개선 필요")
        if reading_speed < 20:
            issues.append("독서 속도 느림")
        issues_text = ", ".join(issues) if issues else "정상적인 시선 패턴"
        
        # 읽기 속도 레벨
        speed_level = "빠름" if reading_speed >= 50 else "적당" if reading_speed >= 20 else "느림"
        concentration_level = "높음" if concentration_score >= 70 else "보통" if concentration_score >= 40 else "주의필요"
        comprehension_level = "매우좋음" if comprehension >= 80 else "보통" if comprehension >= 60 else "연습필요"
        
        # 발음 명확도 레벨
        clarity_level = "명확함" if clarity >= 80 else "괜찮음" if clarity >= 60 else "연습필요"
        fluency_level = "매우좋음" if fluency_score >= 80 else "좋음" if fluency_score >= 60 else "연습필요"
        
        # 종합 피드백
        if concentration_score < 40:
            feedback_text = "집중력 연습이 필요해요! 짧은 시간부터 시작해보세요"
        elif comprehension < 60:
            feedback_text = "이해력을 키우는 연습을 해보세요! 질문하며 읽어보세요"
        elif fluency_score < 60:
            feedback_text = "발음과 말하기 연습을 더 해보세요! 천천히 또박또박"
        else:
            feedback_text = "정말 잘하고 있어요! 계속 꾸준히 읽어보세요"
        
        # 권장 활동
        recommended_activities = []
        if concentration_score < 60:
            recommended_activities.extend(["15분 단위 집중 독서 연습", "시각적 집중력 향상 게임"])
        if not recommended_activities:
            recommended_activities = ["현재 수준 유지", "정기적인 독서 습관"]
        
        # PDF 생성
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        import urllib.request
        import os
        
        # 한글 폰트
        font_path = '/tmp/NotoSansKR.ttf'
        if not os.path.exists(font_path):
            font_url = 'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf'
            urllib.request.urlretrieve(font_url, font_path)
        
        pdfmetrics.registerFont(TTFont('NotoSans', font_path))
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # ===== 헤더 =====
        pdf.setFillColor(colors.HexColor('#667EEA'))
        pdf.rect(0, height-140, width, 140, fill=True, stroke=False)
        
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 26)
        pdf.drawString(50, height-55, "Reading Analysis Report")  # 이모지 제거
        
        pdf.setFont("NotoSans", 13)
        pdf.drawString(50, height-80, f"아동: {child_name}")
        pdf.drawString(50, height-100, f"진단날짜: {datetime.now().strftime('%Y.%m.%d')}")
        pdf.drawString(50, height-120, f"총 읽기시간: {total_time:.0f}초 (집중시간: {focus_time:.0f}초)")
        
        y = height - 180
        
        # ===== 레벨 배지 =====
        level_colors = {
            'advanced': colors.HexColor('#10B981'),
            'intermediate': colors.HexColor('#F59E0B'),
            'beginner': colors.HexColor('#EF4444')
        }
        level_names = {
            'advanced': '우수',
            'intermediate': '보통',
            'beginner': '노력 필요'
        }
        
        level_color = level_colors.get(level_info['level'], colors.grey)
        
        pdf.setFillColor(colors.HexColor('#F9FAFB'))
        pdf.roundRect(200, y-15, 200, 60, 12, fill=True, stroke=False)
        
        pdf.setFillColor(level_color)
        pdf.setFont("NotoSans", 20)
        level_text = level_names.get(level_info['level'], level_info['level'])
        pdf.drawCentredString(300, y+20, f"레벨: {level_text}")
        
        pdf.setFillColor(colors.HexColor('#6B7280'))
        pdf.setFont("NotoSans", 14)
        pdf.drawCentredString(300, y, f"종합 점수: {level_info['total_score']}%")
        
        y -= 100
        
        # ===== 읽기 능력 분석 (확대) =====
        pdf.setFont("NotoSans", 16)
        pdf.setFillColor(colors.HexColor('#1F2937'))
        pdf.drawString(50, y, "읽기 능력 분석")
        y -= 35
        
        # 3개 지표 카드
        scores_data = [
            ("집중력", concentration_score, colors.HexColor('#3B82F6')),
            ("발음 명확도", clarity, colors.HexColor('#8B5CF6')),
            ("유창성", fluency_score, colors.HexColor('#EC4899'))
        ]
        
        card_width = 155
        card_height = 90
        spacing = 15
        
        for i, (label, score, color) in enumerate(scores_data):
            x = 50 + i * (card_width + spacing)
            
            pdf.setFillColor(colors.HexColor('#E5E7EB'))
            pdf.roundRect(x+2, y-card_height-2, card_width, card_height, 10, fill=True, stroke=False)
            
            pdf.setFillColor(colors.white)
            pdf.roundRect(x, y-card_height, card_width, card_height, 10, fill=True, stroke=False)
            
            pdf.setFillColor(color)
            pdf.setFont("Helvetica-Bold", 32)
            pdf.drawCentredString(x + card_width/2, y-38, f"{score:.0f}")
            
            pdf.setFont("Helvetica", 12)
            pdf.drawCentredString(x + card_width/2, y-55, "%")
            
            pdf.setFillColor(colors.HexColor('#4B5563'))
            pdf.setFont("NotoSans", 12)
            pdf.drawCentredString(x + card_width/2, y-75, label)
        
        y -= 130
        
        # ===== 세부 분석 (시선 + 음성) =====
        pdf.setFillColor(colors.HexColor('#F3F4F6'))
        pdf.roundRect(50, y-155, 500, 155, 8, fill=True, stroke=False)
        
        # 시선 추적 결과
        pdf.setFillColor(colors.HexColor('#374151'))
        pdf.setFont("NotoSans", 12)
        pdf.drawString(65, y-25, "시선 추적 결과:")
        pdf.setFont("NotoSans", 10)
        pdf.drawString(75, y-45, f"• 읽기 속도: {speed_level}")
        pdf.drawString(75, y-60, f"• 집중력: {concentration_level}")
        pdf.drawString(75, y-75, f"• 이해력: {comprehension_level}")
        
        # 음성 분석 결과
        pdf.setFont("NotoSans", 12)
        pdf.drawString(300, y-25, "음성 분석 결과:")
        pdf.setFont("NotoSans", 10)
        pdf.drawString(310, y-45, f"• 발음 명확도: {clarity_level} ({clarity:.1f}%)")
        pdf.drawString(310, y-60, f"• 유창성: {fluency_level} ({fluency_score:.1f}%)")
        pdf.drawString(310, y-75, f"• 말하기 속도: {speaking_rate}")
        
        # 시선 패턴
        pdf.setFont("NotoSans", 10)
        pdf.drawString(65, y-105, "시선 패턴:")
        pdf.drawString(75, y-120, issues_text)
        
        # 말한 내용 (축소)
        pdf.drawString(65, y-140, f"말한 내용: {transcription[:40]}...")
        
        y -= 185
        
        # ===== 종합 피드백 =====
        pdf.setFont("NotoSans", 14)
        pdf.setFillColor(colors.HexColor('#1F2937'))
        pdf.drawString(50, y, "종합 피드백")
        y -= 25
        
        pdf.setFillColor(colors.HexColor('#ECFDF5'))
        pdf.roundRect(50, y-40, 500, 40, 8, fill=True, stroke=False)
        
        pdf.setFillColor(colors.HexColor('#065F46'))
        pdf.setFont("NotoSans", 11)
        pdf.drawString(65, y-22, feedback_text)
        
        y -= 65
        
        # ===== 맞춤 추천 활동 =====
        pdf.setFont("NotoSans", 14)
        pdf.setFillColor(colors.HexColor('#1F2937'))
        pdf.drawString(50, y, "맞춤 추천 활동")
        y -= 25
        
        pdf.setFillColor(colors.HexColor('#EFF6FF'))
        pdf.roundRect(50, y-60, 500, 60, 8, fill=True, stroke=False)
        
        pdf.setFillColor(colors.HexColor('#1E40AF'))
        pdf.setFont("NotoSans", 10)
        for idx, activity in enumerate(recommended_activities[:3]):
            pdf.drawString(65, y-20-idx*18, f"• {activity}")
        
        y -= 85
        
        # ===== 하단 정보 =====
        pdf.setStrokeColor(colors.HexColor('#E5E7EB'))
        pdf.setLineWidth(1)
        pdf.line(50, y, 550, y)
        y -= 25
        
        pdf.setFillColor(colors.HexColor('#9CA3AF'))
        pdf.setFont("NotoSans", 9)
        pdf.drawString(50, y, f"이메일: {user_email}")
        next_date = (datetime.now() + timedelta(days=30)).strftime('%Y.%m.%d')
        pdf.drawString(50, y-15, f"다음 검사 추천일: {next_date}")
        pdf.drawRightString(550, y-15, "AI 읽기 진단 시스템")
        
        pdf.save()
        
        pdf_bytes = buffer.getvalue()
        pdf_data = base64.b64encode(pdf_bytes).decode('utf-8')
        
        print("[SUCCESS] PDF 생성 완료")
        
        return jsonify({
            "status": "success",
            "pdf_data": pdf_data,
            "filename": f"{child_name}_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        })
        
    except Exception as e:
        print(f"[ERROR] PDF 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })
# =============================================================================
# 10. Spring 연동 테스트
# =============================================================================

@app.route('/test_spring_connection', methods=['GET'])
def test_spring_connection():
    """Spring 서버 연결 테스트"""
    try:
        import requests
        
        spring_url = "http://15.165.102.27:8080/api/ai/receive"
        
        test_data = {
            "test": "connection",
            "timestamp": datetime.now().isoformat(),
            "message": "Flask에서 보내는 테스트 메시지"
        }
        
        print(f"[INFO] Spring 연결 테스트 시작: {spring_url}")
        
        response = requests.post(
            spring_url,
            json=test_data,
            timeout=5
        )
        
        return jsonify({
            "status": "success",
            "spring_status_code": response.status_code,
            "spring_response": response.text,
            "message": "Spring 서버 연결 성공!"
        })
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "Spring 서버 연결 실패 (서버가 꺼져있거나 주소가 틀렸습니다)"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"테스트 실패: {str(e)}"
        })

# =============================================================================
# 11. 헬스체크
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "calibration_points": len(calibration_data),
        "tracking_results": len(tracking_results)
    })

# =============================================================================
# 12. 서버 실행
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Flask 서버 시작... 포트: {port}")
    print(f"[INFO] 메모리 최적화 모드 활성화")
    app.run(debug=False, host='0.0.0.0', port=port)