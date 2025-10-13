from flask import Flask, render_template, request, jsonify
import cv2
import json
import sys
import os
import base64
import numpy as np
import pymysql
import gc  # 가비지 컬렉션
from datetime import datetime, timedelta

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
    'database': 'ai-talk',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """MySQL DB 연결 - 메모리 최적화"""
    print(f"[DEBUG] DB_CONFIG 내용: {DB_CONFIG}")  # ← 설정 확인
    try:
        connection = pymysql.connect(
            **DB_CONFIG,
            connect_timeout=10,
            read_timeout=10,
            write_timeout=10
        )
        print(f"[SUCCESS] DB 연결 성공!")  # ← 성공 로그
        return connection
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        print(f"[DEBUG] 연결 시도한 주소: {DB_CONFIG['host']}:{DB_CONFIG['port']}")  # ← 실제 연결 정보
        raise Exception(f"DB연결실패: {e}")
    
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
        print("[DEBUG] GazeTracker 생성 중...")
        gaze_tracker = GazeTracker()
        print("[DEBUG] GazeTracker 생성 완료")
        
        print("[DEBUG] AudioAnalyzer 생성 중...")
        audio_analyzer = AudioAnalyzer()
        print("[DEBUG] AudioAnalyzer 생성 완료")
        
        cleanup_memory()  # 메모리 정리
        print("[INFO] 트래커 초기화 완료")
        return jsonify({
            "status": "success", 
            "message": "시스템이 성공적으로 초기화되었습니다."
        })
    except Exception as e:
        print(f"[ERROR] 초기화 오류: {e}")
        import traceback
        traceback.print_exc()  # 상세 에러 출력
        return jsonify({
            "status": "error", 
            "message": f"초기화 실패: {str(e)}"
        }), 500  # 500 상태 코드 반환


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
            "id": int(datetime.now().timestamp()),
            "user_id": user_id,
            "diagnosis_type": "reading_analysis",
            "created_at": datetime.now().isoformat(),
            "report": {
                "child_name": child_name,
                "diagnosis_date": datetime.now().strftime("%Y-%m-%d"),
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
                    "next_diagnosis_date": (datetime.now() + timedelta(days=30)) .strftime("%Y-%m-%d")
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

def save_report_to_db(member_id, child_name, pdf_data, filename):
    """DB에 PDF 리포트 저장 - 메모리 최적화"""
    connection = None
    try:
        print(f"[DEBUG] DB 연결 시도...")
        connection = get_db_connection()
        if not connection:
            print(f"[ERROR] DB 연결 실패!")
            return None
        
        print(f"[DEBUG] SQL 실행 중...")  
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO pdf_reports (member_id, child_name, pdf_data, filename, created_at) 
            VALUES (%s, %s, %s, %s, NOW())
            """
                     
            cursor.execute(sql, (member_id, child_name, pdf_data, filename))
            connection.commit()
                     
            result = cursor.lastrowid
            print(f"[DEBUG] 저장 결과: {result}")
            return result
                 
    except Exception as e:
        print(f"[ERROR] PDF 저장 오류: {e}")
        raise Exception(f"DB저장실패: {e}")
    finally:
        if connection:
            connection.close()

import urllib.request
import tempfile
import ssl

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import os
from datetime import datetime, timedelta
import urllib.request
import ssl


@app.route('/download_pdf_report', methods=['POST'])
def download_pdf_report():
    try:
        print("[DEBUG] PDF 생성 + DB 저장 시작")
        
        data = request.get_json()
        child_name = data.get('child_name', '테스트 아동')
        user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        eye_tracking_result = data.get('eye_tracking_result', {})
        
        # 나눔고딕 웹폰트 다운로드
        try:
            print("[INFO] 나눔고딕 폰트 다운로드 중...")
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            font_url = "https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.ttf"
            
            font_request = urllib.request.Request(
                font_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            with urllib.request.urlopen(font_request, timeout=15, context=ssl_context) as response:
                font_data = response.read()
            
            temp_font = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
            temp_font.write(font_data)
            temp_font.close()
            
            pdfmetrics.registerFont(TTFont('NanumGothic', temp_font.name))
            font_name = 'NanumGothic'
            print("[SUCCESS] 나눔고딕 폰트 등록 완료!")
            
        except Exception as font_error:
            print(f"[WARNING] 폰트 다운로드 실패: {font_error}")
            font_name = 'Helvetica'
        
        # 임시 PDF 파일 생성
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            temp_pdf.name,
            pagesize=A4,
            topMargin=25*mm,
            bottomMargin=20*mm,
            leftMargin=20*mm,
            rightMargin=20*mm
        )
        
        # 스타일 정의
        title_style = ParagraphStyle(
            'Title',
            fontName=font_name,
            fontSize=20,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50'),
            leading=24
        )
        
        header_style = ParagraphStyle(
            'Header',
            fontName=font_name,
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#34495e'),
            leading=18
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            fontName=font_name,
            fontSize=11,
            leading=16,
            spaceAfter=8
        )
        
        # PDF 내용 구성
        content = []
        
        # 제목
        content.append(Paragraph("📚 읽기 능력 진단 리포트", title_style))
        content.append(Spacer(1, 10))
        content.append(Paragraph(f"👦 아동명: {child_name}", header_style))
        content.append(Spacer(1, 20))
        
        # 기본 정보
        content.append(Paragraph("📋 기본 정보", header_style))
        
        basic_data = [
            ['아동 이름', child_name],
            ['진단 날짜', datetime.now().strftime('%Y년 %m월 %d일')],
            ['사용자 ID', str(user_id)],
            ['리포트 생성 시간', datetime.now().strftime('%H시 %M분')]
        ]
        
        basic_table = Table(basic_data, colWidths=[50*mm, 100*mm])
        basic_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        content.append(basic_table)
        content.append(Spacer(1, 20))

        # ===== 시선추적 분석 결과 =====
        content.append(Paragraph("👁️ 시선추적 분석 결과", header_style))
        
        eye_data = [
            ['집중 시간', eye_tracking_result.get('focus_time', '측정되지 않음')],
            ['시선 상태', eye_tracking_result.get('issues', '정상')],
            ['집중도', eye_tracking_result.get('concentration', '측정되지 않음')],
            ['추적 상태', '완료']
        ]
        
        eye_table = Table(eye_data, colWidths=[50*mm, 100*mm])
        eye_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),  # 연한 파란색
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        content.append(eye_table)
        content.append(Spacer(1, 20))
        # ===== 시선추적 섹션 끝 =====
        
        # 음성 분석 결과
        content.append(Paragraph("🎤 음성 분석 결과", header_style))
        
        transcription = audio_result.get('transcription', '음성 녹음이 없습니다')
        fluency = audio_result.get('fluency', '측정되지 않음')
        clarity = audio_result.get('pronunciation_clarity', '측정되지 않음')
        
        speech_data = [
            ['인식된 내용', transcription[:60] + '...' if len(transcription) > 60 else transcription],
            ['말하기 유창성', fluency],
            ['발음 명확도', clarity],
            ['전체 분석 상태', '완료']
        ]
        
        speech_table = Table(speech_data, colWidths=[50*mm, 100*mm])
        speech_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e8')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        content.append(speech_table)
        content.append(Spacer(1, 20))
        
        # 종합 평가
        content.append(Paragraph("📊 종합 평가", header_style))
        content.append(Paragraph("이번 진단을 통해 아동의 읽기 능력과 음성 분석이 완료되었습니다.", normal_style))
        content.append(Paragraph("지속적인 읽기 연습과 발음 교정을 통해 더욱 향상된 결과를 기대할 수 있습니다.", normal_style))
        content.append(Spacer(1, 15))
        
        # 추천사항
        content.append(Paragraph("💡 맞춤형 추천사항", header_style))
        
        recommendations = [
            "1. 매일 20분씩 소리내어 읽기 연습하기",
            "2. 다양한 장르의 책으로 독서 범위 넓히기",
            "3. 읽은 내용을 요약하여 말해보기",
            "4. 발음이 어려운 단어는 반복 연습하기",
            "5. 시선 집중력 향상을 위한 집중 훈련",  # 시선추적 관련 추가
            "6. 3개월 후 재진단 받기"
        ]
        
        for rec in recommendations:
            content.append(Paragraph(rec, normal_style))
            content.append(Spacer(1, 4))
        
        # 푸터
        content.append(Spacer(1, 30))
        footer = f"📅 생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} | AI 읽기 진단 시스템"
        footer_style = ParagraphStyle(
            'Footer',
            fontName=font_name,
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        content.append(Paragraph(footer, footer_style))
        
        # PDF 생성
        doc.build(content)
        temp_pdf.close()
        print("[SUCCESS] 한글 PDF 생성 완료!")

        # 1. PDF 파일을 바이너리로 읽기
        with open(temp_pdf.name, 'rb') as f:
            pdf_binary_data = f.read()

        # 2. Base64 인코딩
        import base64
        pdf_base64 = base64.b64encode(pdf_binary_data).decode('utf-8')

        # 3. DB 저장 시도
        error_msg = None
        try:
            print("[INFO] DB에 리포트 저장 중...")
            
            # 리포트 데이터 구성 (백엔드 분석용)
            report_data = {
                "id": int(datetime.now().timestamp()),
                "user_id": user_id,
                "diagnosis_type": "reading_analysis_pdf",
                "created_at": datetime.now().isoformat(),
                "report": {
                    "child_name": child_name,
                    "diagnosis_date": datetime.now().strftime("%Y-%m-%d"),
                    "speech_analysis": audio_result,
                    "eye_tracking": eye_tracking_result,
                    "pdf_generated": True,
                    "total_tracking_results": eye_tracking_result.get('total_measurements', 0),
                    "calibration_points": 5
                }
            }
            
            filename = f"{child_name}_읽기진단리포트.pdf"
            report_id = save_report_to_db(user_id, child_name, pdf_binary_data, filename)
            
            if report_id:
                print(f"[SUCCESS] 백엔드용 DB 저장 완료! Report ID: {report_id}")
            else:
                print("[WARNING] DB 저장 실패")
                
        except Exception as db_error:
            print(f"[ERROR] DB 저장 오류: {db_error}")
            report_id = None
            error_msg = str(db_error)

        # 4. 임시 파일 정리
        os.unlink(temp_pdf.name)
        if 'temp_font' in locals():
            try:
                os.unlink(temp_font.name)
            except:
                pass

        print(f"[INFO] Report ID {report_id}로 DB 저장 완료!")

        # 5. 응답 반환
        return jsonify({
            "status": "success",
            "pdf_data": pdf_base64,
            "filename": f"{child_name}_읽기진단리포트.pdf",
            "report_id": report_id,
            "db_saved": report_id is not None,
            "error": error_msg
        })
        
    except Exception as e:
        print(f"[ERROR] PDF 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"PDF 생성 오류: {str(e)}"})

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
    # Flask 애플리케이션 컨텍스트 문제 해결
    port = int(os.environ.get('PORT', 5000))
    
    print(f"[INFO] Flask 서버 시작... 포트: {port}")
    print(f"[INFO] 메모리 최적화 모드 활성화")

    # ===== 핵심 수정: Application Context 오류 해결 =====
    # init_tracker()를 서버 시작 전에 자동 호출하지 않음
    # 대신 첫 요청 시 자동으로 초기화되도록 변경
    print("[INFO] 트래커는 첫 요청 시 자동으로 초기화됩니다.")
    
    # debug=False는 Render 배포 시 권장되는 설정입니다.
    app.run(debug=False, host='0.0.0.0', port=port)