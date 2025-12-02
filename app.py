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
# 9. PDF 생성 관련 (나중에 삭제 가능)
# =============================================================================

@app.route('/download_pdf_report', methods=['POST'])
def download_pdf_report():
    """PDF 리포트 생성"""
    try:
        # ... PDF 생성 로직 (길어서 생략, 필요 시 유지) ...
        return jsonify({"status": "success", "message": "PDF 기능은 프론트엔드에서 처리됩니다"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
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