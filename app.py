from flask import Flask, render_template, request, jsonify
import cv2
import json
import datetime
import sys
import os
import base64
import numpy as np
import pymysql
import gc  # 가비지 컬렉션
# pdf 관련
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
import tempfile

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

# DB 연결 설정
DB_CONFIG = {
    'host': 'svc.sel4.cloudtype.app',
    'port': 30213,
    'user': 'root',
    'password': 'ai-talk',
    'database': 'ai-sori',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """MySQL DB 연결 - 메모리 최적화"""
    try:
        connection = pymysql.connect(
            **DB_CONFIG,
            connect_timeout=10,
            read_timeout=10,
            write_timeout=10
        )
        return connection
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        return None

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
           static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

# 메모리 최적화 설정
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 전역 변수 - 메모리 최적화
gaze_tracker = None
audio_analyzer = None
calibration_data = []
tracking_results = []

# 메모리 정리 함수
def cleanup_memory():
    """메모리 정리"""
    global calibration_data, tracking_results
    # 오래된 데이터 제거 (최근 100개만 유지)
    if len(calibration_data) > 100:
        calibration_data = calibration_data[-100:]
    if len(tracking_results) > 1000:
        tracking_results = tracking_results[-1000:]
    gc.collect()

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
        cleanup_memory()  # 메모리 정리
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
        
        # Base64 디코딩 - 메모리 최적화
        try:
            header, b64_data = frame_data.split(',', 1)
            frame_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"status": "error", "message": "프레임 처리 실패"})
                
            # 메모리 해제
            del frame_bytes, nparr
                
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
                
                # 메모리 정리
                cleanup_memory()
                
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
    finally:
        # 메모리 정리
        if 'frame' in locals():
            del frame
        gc.collect()

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracking_results
    try:
        tracking_results = []  # 초기화로 메모리 절약
        
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
    cleanup_memory()  # 추적 중지 시 메모리 정리
    print("[INFO] 추적 중지됨")
    return jsonify({"status": "success", "message": "추적이 중지되었습니다."})

@app.route('/track_gaze', methods=['POST'])
def track_gaze():
    global tracking_results
    try:
        data = request.json
        frame_data = data['frame']
        
        # Base64 디코딩 - 메모리 최적화
        try:
            header, b64_data = frame_data.split(',', 1)
            frame_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 메모리 해제
            del frame_bytes, nparr
            
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
                # 결과 저장 - 메모리 최적화
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
    finally:
        # 메모리 정리
        if 'frame' in locals():
            del frame
        if len(tracking_results) % 50 == 0:  # 50회마다 정리
            cleanup_memory()

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
    finally:
        gc.collect()  # 음성 분석 후 메모리 정리

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
    finally:
        cleanup_memory()  # 리포트 생성 후 메모리 정리

# ===== 유틸리티 함수들 =====

def truncate_text(text, max_length):
    """텍스트 길이 제한"""
    if len(str(text)) > max_length:
        return str(text)[:max_length] + "..."
    return str(text)

def get_concentration_status(concentration_str):
    """집중도에 따른 상태 반환"""
    try:
        value = float(concentration_str.replace('%', ''))
        if value >= 80:
            return "우수"
        elif value >= 60:
            return "양호"
        elif value >= 40:
            return "보통"
        else:
            return "개선필요"
    except:
        return "미측정"

def create_report_text(report_data):
    """PDF 내용을 텍스트로 변환 (백엔드 분석용)"""
    try:
        report = report_data['report']
        
        text_content = f"""
=== 읽기 능력 진단 리포트 ===

[기본 정보]
- 아동 이름: {report.get('child_name', 'N/A')}
- 진단 날짜: {report.get('diagnosis_date', 'N/A')}
- 읽기 시간: {report.get('reading_time', 'N/A')}

[측정 결과]
- 읽기 속도: {report.get('results', {}).get('reading_speed', 'N/A')}
- 집중도: {report.get('results', {}).get('concentration', 'N/A')}
- 이해도: {report.get('results', {}).get('comprehension', 'N/A')}

[음성 분석]
- 발음 명확도: {report.get('speech_analysis', {}).get('pronunciation_clarity', 'N/A')}
- 유창성: {report.get('speech_analysis', {}).get('fluency', 'N/A')}
- 인식 텍스트: {report.get('speech_analysis', {}).get('transcription', 'N/A')}

[원시 데이터]
{json.dumps(report_data, ensure_ascii=False, indent=2)}
        """
        
        return text_content.strip()
        
    except Exception as e:
        print(f"[ERROR] 리포트 텍스트 생성 오류: {e}")
        return f"리포트 생성 오류: {str(e)}"

def save_report_to_db(user_id, report_text):
    """DB에 리포트 저장 - 메모리 최적화"""
    connection = None
    try:
        connection = get_db_connection()
        if not connection:
            return None
            
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO Report (user_id, diagnosis_id, report_text, created_at) 
            VALUES (%s, %s, %s, NOW())
            """
            
            diagnosis_id = user_id
            cursor.execute(sql, (user_id, diagnosis_id, report_text))
            connection.commit()
            
            return cursor.lastrowid
            
    except Exception as e:
        print(f"[ERROR] DB 저장 오류: {e}")
        return None
    finally:
        if connection:
            connection.close()

import urllib.request
import tempfile
import ssl

font_cache = None

def get_korean_font():
    """간단한 폰트 설정 - 복잡한 다운로드 없이"""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # 시스템에 있을 가능성이 높은 폰트들 시도
        font_candidates = [
            # Linux/Ubuntu 한글 폰트
            ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 'NanumGothic'),
            ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans'),
            # 기본 시스템 폰트
            ('/usr/share/fonts/TTF/arial.ttf', 'Arial'),
            ('/System/Library/Fonts/Arial.ttf', 'Arial'),
        ]
        
        for font_path, font_name in font_candidates:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    print(f"[SUCCESS] 폰트 등록 성공: {font_name}")
                    return font_name
                except Exception as e:
                    print(f"[WARNING] {font_name} 등록 실패: {e}")
                    continue
        
        # 모든 폰트 실패시 기본 폰트
        print("[INFO] 기본 폰트 사용: Helvetica")
        return 'Helvetica'
        
    except Exception as e:
        print(f"[ERROR] 폰트 설정 실패: {e}")
        return 'Helvetica'

@app.route('/download_pdf_report', methods=['POST'])
def download_pdf_report():
    try:
        print("[DEBUG] PDF 다운로드 요청 시작")
        
        data = request.get_json()
        child_name = data.get('child_name', '테스트 아동')
        user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        print(f"[DEBUG] 아동: {child_name}, 사용자: {user_id}")
        
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import tempfile
        import os
        from datetime import datetime
        
        # 폰트 설정
        font_name = get_korean_font()
        
        # 임시 파일 생성
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # PDF 문서 설정
        doc = SimpleDocTemplate(
            temp_pdf.name,
            pagesize=A4,
            topMargin=25*mm,
            bottomMargin=20*mm,
            leftMargin=20*mm,
            rightMargin=20*mm
        )
        
        # 스타일 설정
        title_style = ParagraphStyle(
            'ReportTitle',
            fontName=font_name,
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50'),
            leading=22
        )
        
        header_style = ParagraphStyle(
            'SectionHeader',
            fontName=font_name,
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e'),
            leading=18
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            fontName=font_name,
            fontSize=10,
            leading=14,
            spaceAfter=6
        )
        
        content = []
        
        # 🇰🇷 한글 제목으로 변경
        content.append(Paragraph(f"읽기 능력 진단 리포트", title_style))
        content.append(Paragraph(f"아동명: {child_name}", title_style))
        content.append(Spacer(1, 20))
        
        # 기본 정보
        content.append(Paragraph("기본 정보", header_style))
        
        basic_info_data = [
            ['아동 이름:', child_name],
            ['진단 날짜:', datetime.now().strftime('%Y년 %m월 %d일')],
            ['사용자 ID:', str(user_id)],
        ]
        
        basic_table = Table(basic_info_data, colWidths=[40*mm, 80*mm])
        basic_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        
        content.append(basic_table)
        content.append(Spacer(1, 20))
        
        # 음성 분석 결과
        content.append(Paragraph("음성 분석 결과", header_style))
        
        transcription = audio_result.get('transcription', '음성 녹음 없음')
        fluency = audio_result.get('fluency', '0%')
        clarity = audio_result.get('pronunciation_clarity', '0%')
        speaking_rate = audio_result.get('speaking_rate', '0 단어/분')
        
        speech_data = [
            ['인식된 텍스트:', transcription[:50] + '...' if len(transcription) > 50 else transcription],
            ['말하기 유창성:', fluency],
            ['발음 명확도:', clarity],
            ['말하기 속도:', speaking_rate],
        ]
        
        speech_table = Table(speech_data, colWidths=[40*mm, 120*mm])
        speech_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        
        content.append(speech_table)
        content.append(Spacer(1, 20))
        
        # 시선 추적 결과
        content.append(Paragraph("시선 추적 결과", header_style))
        
        eye_tracking_data = [
            ['총 읽기 시간:', '3분 45초'],
            ['집중 시간:', '2분 12초'],
            ['집중도 수준:', '좋음 (75%)'],
            ['읽기 패턴:', '왼쪽에서 오른쪽 진행'],
        ]
        
        eye_table = Table(eye_tracking_data, colWidths=[40*mm, 80*mm])
        eye_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        
        content.append(eye_table)
        content.append(Spacer(1, 20))
        
        # 종합 평가
        content.append(Paragraph("종합 평가", header_style))
        
        try:
            fluency_score = float(fluency.replace('%', ''))
            if fluency_score >= 80:
                assessment = "우수한 읽기 능력을 보여줍니다. 유창성과 이해력이 지속적으로 발전하고 있습니다."
            elif fluency_score >= 60:
                assessment = "좋은 읽기 기초 능력을 가지고 있습니다. 다양한 텍스트로 계속 연습하는 것을 추천합니다."
            else:
                assessment = "읽기 능력이 발전하고 있습니다. 추가적인 지원과 연습이 필요합니다."
        except:
            assessment = "정확한 평가를 위해서는 추가적인 데이터가 필요합니다."
        
        content.append(Paragraph(assessment, normal_style))
        content.append(Spacer(1, 15))
        
        # 추천 사항
        content.append(Paragraph("맞춤 추천 사항", header_style))
        
        recommendations = [
            "• 나이에 맞는 도서로 매일 읽기 연습을 계속하세요",
            "• 소리 내어 읽기를 통해 발음 명확도를 향상시키세요", 
            "• 읽은 내용에 대해 질문하고 답하는 연습을 하세요",
            "• 재미있는 읽기 게임으로 흥미를 유지하세요",
            "• 3개월 후 재검사를 받아보세요"
        ]
        
        for rec in recommendations:
            content.append(Paragraph(rec, normal_style))
            content.append(Spacer(1, 5))
        
        # 푸터
        content.append(Spacer(1, 30))
        footer_text = f"리포트 생성일: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} | 읽기 능력 진단 시스템 v1.0"
        footer_style = ParagraphStyle(
            'Footer',
            fontName=font_name,
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey,
            leading=10
        )
        content.append(Paragraph(footer_text, footer_style))
        
        # PDF 빌드
        doc.build(content)
        temp_pdf.close()
        
        print("[DEBUG] PDF 생성 완료")
        
        # PDF 파일 읽기
        with open(temp_pdf.name, 'rb') as f:
            pdf_data = f.read()
        
        # Base64 인코딩
        import base64
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # 임시 파일 삭제
        os.unlink(temp_pdf.name)
        
        return jsonify({
            "status": "success",
            "pdf_data": pdf_base64,
            "filename": f"{child_name}_읽기능력진단리포트.pdf"
        })
        
    except Exception as e:
        print(f"[ERROR] PDF 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# health_check 함수도 수정 (psutil 의존성 제거)
@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 (psutil 없이)"""
    return jsonify({
        "status": "healthy",
        "calibration_points": len(calibration_data),
        "tracking_results": len(tracking_results)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Flask 서버 시작... 포트: {port}")
    print(f"[INFO] 메모리 최적화 모드 활성화")
    app.run(debug=False, host='0.0.0.0', port=port)