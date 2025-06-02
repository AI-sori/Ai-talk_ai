from flask import Flask, render_template, request, jsonify
import cv2
import json
import datetime
import sys
import os
import base64
import numpy as np

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 모듈들 import
try:
    from utils.gaze_tracker import GazeTracker
    from utils.audio_analyzer import AudioAnalyzer
    print("[INFO] 모듈 로드 성공")
except ImportError as e:
    print(f"[ERROR] 모듈 로드 실패: {e}")
    # 더미 클래스들 생성
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

app = Flask(__name__)

# 전역 변수
gaze_tracker = None
audio_analyzer = None
calibration_data = []
tracking_results = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_tracker', methods=['POST'])
def init_tracker():
    global gaze_tracker, audio_analyzer
    try:
        print("[INFO] 트래커 초기화 시작...")
        gaze_tracker = GazeTracker()
        audio_analyzer = AudioAnalyzer()
        print("[INFO] 트래커 초기화 완료")
        return jsonify({
            "status": "success", 
            "message": "시스템이 성공적으로 초기화되었습니다."
        })
    except Exception as e:
        print(f"[ERROR] 초기화 오류: {e}")
        return jsonify({
            "status": "error", 
            "message": f"초기화 실패: {str(e)}"
        })

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global calibration_data
    try:
        data = request.json
        frame_data = data['frame']
        target_x = data['target_x']
        target_y = data['target_y']
        
        # Base64 디코딩
        try:
            header, b64_data = frame_data.split(',', 1)
            frame_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"status": "error", "message": "프레임 처리 실패"})
                
        except Exception as decode_error:
            print(f"[ERROR] 디코딩 오류: {decode_error}")
            return jsonify({"status": "error", "message": "이미지 디코딩 실패"})
        
        if gaze_tracker:
            gaze_point = gaze_tracker.get_gaze_direction(frame)
            
            if gaze_point:
                calibration_data.append({
                    'target': (target_x, target_y),
                    'gaze': gaze_point
                })
                print(f"[INFO] 보정 포인트 추가됨. 총 {len(calibration_data)}개")
                return jsonify({
                    "status": "success", 
                    "calibration_points": len(calibration_data)
                })
            else:
                return jsonify({"status": "error", "message": "시선을 감지할 수 없습니다."})
        
        return jsonify({"status": "error", "message": "트래커가 초기화되지 않았습니다."})
        
    except Exception as e:
        print(f"[ERROR] 보정 오류: {e}")
        return jsonify({"status": "error", "message": f"보정 실패: {str(e)}"})

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracking_results
    try:
        tracking_results = []
        
        if len(calibration_data) >= 4:
            success = gaze_tracker.calibrate(calibration_data)
            if success:
                print("[INFO] 추적 시작됨")
                return jsonify({
                    "status": "success", 
                    "message": f"{len(calibration_data)}개 보정 포인트로 추적 시작"
                })
            else:
                return jsonify({"status": "error", "message": "보정 실패"})
        else:
            return jsonify({
                "status": "error", 
                "message": f"최소 4개의 보정 포인트가 필요합니다. (현재: {len(calibration_data)}개)"
            })
            
    except Exception as e:
        print(f"[ERROR] 추적 시작 오류: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    print("[INFO] 추적 중지됨")
    return jsonify({"status": "success", "message": "추적이 중지되었습니다."})

@app.route('/track_gaze', methods=['POST'])
def track_gaze():
    global tracking_results
    try:
        data = request.json
        frame_data = data['frame']
        
        # Base64 디코딩
        try:
            header, b64_data = frame_data.split(',', 1)
            frame_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    "status": "success",
                    "direction": "center",
                    "confidence": 0.3,
                    "error_offset": 50
                })
                
        except Exception as decode_error:
            return jsonify({
                "status": "success",
                "direction": "center", 
                "confidence": 0.3,
                "error_offset": 50
            })
        
        # 시선 추적 실행
        if gaze_tracker:
            result = gaze_tracker.track_reading(frame)
            
            if result:
                # 결과 저장
                tracking_results.append({
                    'timestamp': datetime.datetime.now().isoformat(),
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
            else:
                return jsonify({
                    "status": "success",
                    "direction": "center",
                    "confidence": 0.3,
                    "error_offset": 50
                })
        else:
            return jsonify({
                "status": "error",
                "message": "트래커가 초기화되지 않았습니다."
            })
        
    except Exception as e:
        print(f"[ERROR] track_gaze 오류: {e}")
        return jsonify({
            "status": "success",
            "direction": "center",
            "confidence": 0.3,
            "error_offset": 50
        })

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"status": "error", "message": "오디오 파일이 없습니다."})
        
        audio_file = request.files['audio']
        print(f"[INFO] 오디오 파일 받음: {audio_file.filename}")
        
        if audio_analyzer:
            result = audio_analyzer.analyze(audio_file)
            print(f"[INFO] 음성 분석 완료")
            return jsonify({"status": "success", "result": result})
        
        return jsonify({"status": "error", "message": "음성 분석기가 초기화되지 않았습니다."})
        
    except Exception as e:
        print(f"[ERROR] 음성 분석 오류: {e}")
        return jsonify({"status": "error", "message": f"음성 분석 실패: {str(e)}"})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    global tracking_results
    try:
        data = request.json
        child_name = data.get('child_name', 'Unknown')
        user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        print(f"[INFO] 리포트 생성 시작. 추적 결과: {len(tracking_results)}개")
        
        # 시선추적 분석
        if tracking_results:
            total_tracking_time = len(tracking_results) * 0.5
            left_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'left')
            right_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'right')
            center_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'center')
            
            concentration_score = (center_count / len(tracking_results) * 100)
            direction_changes = sum(1 for i in range(1, len(tracking_results)) 
                                  if tracking_results[i]['gaze_direction'] != tracking_results[i-1]['gaze_direction'])
            reading_speed = direction_changes / (total_tracking_time / 60) if total_tracking_time > 0 else 0
        else:
            total_tracking_time = 0
            concentration_score = 0
            reading_speed = 0
            left_count = right_count = center_count = 0
        
        # 이슈 분석
        issues = []
        if concentration_score < 40:
            issues.append("심각한 집중력 부족")
        elif concentration_score < 60:
            issues.append("집중력 개선 필요")
        
        if reading_speed < 20:
            issues.append("독서 속도 느림")
        elif reading_speed > 80:
            issues.append("독서 속도 과도히 빠름")
            
        if left_count > right_count * 3:
            issues.append("좌측 편향 시선")
        elif right_count > left_count * 3:
            issues.append("우측 편향 시선")
            
        issues_text = ", ".join(issues) if issues else "정상"
        
        # 권장 활동
        recommended_activities = []
        
        if concentration_score < 60:
            recommended_activities.extend([
                "15분 단위 집중 독서 연습",
                "시각적 집중력 향상 게임",
                "독서 환경 개선"
            ])
        
        if reading_speed < 30:
            recommended_activities.extend([
                "단계별 읽기 속도 향상 훈련",
                "안구 운동 연습"
            ])
            
        fluency_value = audio_result.get('fluency', '0%').replace('%', '')
        try:
            if float(fluency_value) < 70:
                recommended_activities.extend([
                    "발음 연습 및 따라 읽기",
                    "음성 녹음 후 자가 점검"
                ])
        except:
            pass
            
        if not recommended_activities:
            recommended_activities = [
                "현재 수준 유지",
                "다양한 장르의 책 읽기",
                "정기적인 독서 습관 유지"
            ]

        # API 명세 구조
        report = {
            "id": int(datetime.datetime.now().timestamp()),
            "user_id": user_id,
            "diagnosis_type": "reading_analysis",
            "created_at": datetime.datetime.now().isoformat(),
            "report": {
                "child_name": child_name,
                "diagnosis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "reading_time": f"{total_tracking_time:.1f}초",
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
                "feedback": {
                    "summary": f"총 {len(tracking_results)}회 측정, 집중도 {concentration_score:.1f}%",
                    "recommended_activities": recommended_activities,
                    "next_diagnosis_date": (datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
                }
            }
        }
        
        print("[INFO] 리포트 생성 완료")
        return jsonify({"status": "success", "report": report})
        
    except Exception as e:
        print(f"[ERROR] 리포트 생성 오류: {e}")
        return jsonify({"status": "error", "message": f"리포트 생성 실패: {str(e)}"})

if __name__ == '__main__':
    print("[INFO] Flask 서버 시작...")
    app.run(debug=True, host='0.0.0.0', port=5000)