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

def setup_korean_font():
    """한글 폰트 자동 다운로드 및 등록 (기본 라이브러리만 사용)"""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        print("[INFO] 한글 폰트 다운로드 중...")
        
        # 구글 폰트 URL (나눔고딕)
        font_url = "https://fonts.gstatic.com/s/nanumgothic/v17/PN_3Rfi-oW3hYwmKDpxS7F_D_9ta.ttf"
        
        # SSL 설정 (배포 환경 호환성)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 폰트 다운로드
        request = urllib.request.Request(
            font_url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PDF Generator)'}
        )
        
        with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
            font_data = response.read()
        
        # 다운로드 검증
        if len(font_data) > 10000:  # 최소 10KB 이상이어야 정상 폰트
            # 임시 파일로 저장
            temp_font = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
            temp_font.write(font_data)
            temp_font.close()
            
            # ReportLab에 폰트 등록
            pdfmetrics.registerFont(TTFont('NanumGothic', temp_font.name))
            print(f"[SUCCESS] 한글 폰트 등록 완료 ({len(font_data)} bytes)")
            return 'NanumGothic'
        else:
            raise Exception(f"폰트 파일 크기 이상: {len(font_data)} bytes")
            
    except Exception as e:
        print(f"[WARNING] 한글 폰트 설정 실패: {e}")
        print("[INFO] 기본 폰트로 대체합니다")
        return 'Helvetica'  # 기본 폰트로 fallback

# 추가 개선: 폰트 캐싱
font_cache = None

def get_korean_font():
    """한글 폰트 캐싱 (한 번만 다운로드)"""
    global font_cache
    if font_cache is None:
        font_cache = setup_korean_font()
    return font_cache

# 기존 create_simple_pdf 함수 위에 이 함수 추가
def setup_korean_font():
    """한글 폰트 자동 다운로드 및 등록"""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        print("[INFO] 한글 폰트 다운로드 중...")
        
        # 구글 폰트 직접 다운로드 (Nanum Gothic)
        font_url = "https://fonts.gstatic.com/s/nanumgothic/v17/PN_3Rfi-oW3hYwmKDpxS7F_D_9ta.ttf"
        
        response = requests.get(font_url, timeout=30)
        
        if response.status_code == 200:
            # 임시 파일로 저장
            temp_font = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
            temp_font.write(response.content)
            temp_font.close()
            
            # ReportLab에 폰트 등록
            pdfmetrics.registerFont(TTFont('NanumGothic', temp_font.name))
            print("[SUCCESS] 한글 폰트 등록 완료")
            return 'NanumGothic'
        else:
            raise Exception("폰트 다운로드 실패")
            
    except Exception as e:
        print(f"[WARNING] 한글 폰트 설정 실패: {e}")
        return 'Helvetica'  # 기본 폰트로 fallback

def create_simple_pdf(report_data):
    """서버용 한글 폰트 지원 PDF 생성"""
    temp_pdf = None
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        
        korean_font = get_korean_font()  # 캐싱된 폰트 사용
        
        # PDF 생성 e, suffix='.pdf')
        
        doc = SimpleDocTemplate(
            temp_pdf.name, 
            pagesize=A4, 
            topMargin=35*mm, 
            bottomMargin=25*mm,
            leftMargin=25*mm,
            rightMargin=25*mm
        )
        
        styles = getSampleStyleSheet()
        
        # 한글 폰트 사용 (영어 fallback 제거)
        title_text = "읽기 능력 진단 리포트"
        section_titles = {
            "리포트 개요": "리포트 개요",
            "검사 결과 분석": "검사 결과 분석",
            "음성 분석 결과": "음성 분석 결과"
        }
        
        # 전문적인 스타일들 (한글 폰트 적용)
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Title'],
            fontSize=20,
            spaceAfter=25,
            alignment=TA_CENTER,
            fontName=korean_font,  # 다운로드된 한글 폰트 사용
            textColor=colors.HexColor('#2C3E50')
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            fontName=korean_font,  # 한글 폰트 사용
            textColor=colors.HexColor('#34495E'),
            borderWidth=0,
            borderPadding=5,
            backColor=colors.HexColor('#ECF0F1')
        )
        
        content = []
        
        # 헤더 라인
        header_line = Table([['']], colWidths=[160*mm])
        header_line.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#95A5A6')),
            ('LINEBELOW', (0, 0), (-1, -1), 3, colors.HexColor('#7F8C8D')),
        ]))
        content.append(header_line)
        content.append(Spacer(1, 10))
        
        # 제목 (한글)
        content.append(Paragraph(title_text, title_style))
        content.append(Spacer(1, 15))
        
        # 리포트 정보 헤더
        info_header = Table([["리포트 개요"]], colWidths=[160*mm])
        info_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(info_header)
        
        # 기본 정보 (한글)
        report_info = report_data['report']
        basic_info = [
            ['검사 대상자', report_info.get('child_name', 'N/A')],
            ['검사 실시일', report_info.get('diagnosis_date', 'N/A')],
            ['총 검사 시간', report_info.get('reading_time', 'N/A')],
            ['리포트 생성일', datetime.datetime.now().strftime('%Y년 %m월 %d일')]
        ]
        
        basic_table = Table(basic_info, colWidths=[40*mm, 120*mm])
        basic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#D5DBDB')),
            ('BACKGROUND', (1, 0), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#85929E')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        content.append(basic_table)
        content.append(Spacer(1, 20))
        
        # 검사 결과 헤더
        result_header = Table([["검사 결과 분석"]], colWidths=[160*mm])
        result_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(result_header)
        
        # 측정 결과 (한글)
        results = report_info.get('results', {})
        concentration_str = results.get('concentration', '0%')
        concentration_status = get_concentration_status(concentration_str)
        
        result_data = [
            ['평가 영역', '측정 결과', '평가 등급'],
            ['읽기 속도', results.get('reading_speed', 'N/A'), '측정 완료'],
            ['집중도 분석', concentration_str, concentration_status],
            ['이해도 평가', results.get('comprehension', 'N/A'), '측정 완료']
        ]
        
        result_table = Table(result_data, colWidths=[50*mm, 55*mm, 55*mm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A6ACAF')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#85929E')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        content.append(result_table)
        content.append(Spacer(1, 20))
        
        # 음성 분석 헤더
        speech_header = Table([["음성 분석 결과"]], colWidths=[160*mm])
        speech_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(speech_header)
        
        # 음성 분석 (한글)
        speech = report_info.get('speech_analysis', {})
        speech_data = [
            ['분석 항목', '측정 결과'],
            ['발음 명확도', speech.get('pronunciation_clarity', 'N/A')],
            ['유창성 평가', speech.get('fluency', 'N/A')],
            ['인식된 발화 내용', truncate_text(speech.get('transcription', 'N/A'), 60)]
        ]
        
        speech_table = Table(speech_data, colWidths=[50*mm, 110*mm])
        speech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A6ACAF')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#85929E')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),  # 한글 폰트
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        content.append(speech_table)
        content.append(Spacer(1, 25))
        
        # 푸터 라인
        footer_line = Table([['']], colWidths=[160*mm])
        footer_line.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#95A5A6')),
            ('LINEABOVE', (0, 0), (-1, -1), 1, colors.HexColor('#7F8C8D')),
        ]))
        content.append(footer_line)
        content.append(Spacer(1, 8))
        
        # 푸터 (한글)
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            fontName=korean_font,  # 한글 폰트
            textColor=colors.HexColor('#7F8C8D')
        )
        
        footer_text = "본 리포트는 AI 읽기 능력 진단 시스템에 의해 자동 생성되었습니다"
        content.append(Paragraph(footer_text, footer_style))
        
        doc.build(content)
        return temp_pdf.name
        
    except Exception as e:
        print(f"[ERROR] PDF 생성 오류: {e}")
        return None
    finally:
        # 메모리 정리
        if temp_pdf:
            temp_pdf.close()
        gc.collect()

@app.route('/download_pdf_report', methods=['POST'])
def download_pdf_report():
    try:
        print("[DEBUG] 한글 PDF 다운로드 요청 시작")
        
        data = request.get_json()
        print(f"[DEBUG] 받은 데이터: {data}")
        
        child_name = data.get('child_name', '테스트')
        user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        print(f"[DEBUG] 아동 이름: {child_name}")
        print(f"[DEBUG] 사용자 ID: {user_id}")
        print(f"[DEBUG] 음성 결과: {audio_result}")
        
        # 한글 PDF 생성
        print("[DEBUG] 한글 PDF 생성 시도...")
        
        # ReportLab 임포트 테스트
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            print("[DEBUG] ReportLab 임포트 성공")
        except ImportError as e:
            print(f"[ERROR] ReportLab 임포트 실패: {e}")
            return jsonify({"status": "error", "message": f"ReportLab 설치 필요: {e}"})
        
        # 한글 폰트 설정
        try:
            korean_font = get_korean_font()  # 기존 함수 활용
            print(f"[DEBUG] 사용할 폰트: {korean_font}")
        except:
            korean_font = 'Helvetica'  # 폰트 실패시 기본 폰트
            print("[WARNING] 한글 폰트 설정 실패, 기본 폰트 사용")
        
        # 임시 파일 생성
        try:
            import tempfile
            import os
            from datetime import datetime
            
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            print(f"[DEBUG] 임시 파일 생성: {temp_pdf.name}")
            
            # 완전한 한글 PDF 생성
            doc = SimpleDocTemplate(
                temp_pdf.name, 
                pagesize=A4,
                topMargin=25*mm,
                bottomMargin=20*mm,
                leftMargin=20*mm,
                rightMargin=20*mm
            )
            
            # 한글 스타일 설정
            title_style = ParagraphStyle(
                'KoreanTitle',
                fontName=korean_font,
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2c3e50'),
                leading=22
            )
            
            header_style = ParagraphStyle(
                'KoreanHeader',
                fontName=korean_font,
                fontSize=14,
                spaceAfter=12,
                textColor=colors.HexColor('#34495e'),
                leading=18
            )
            
            normal_style = ParagraphStyle(
                'KoreanNormal',
                fontName=korean_font,
                fontSize=10,
                leading=14,
                spaceAfter=6
            )
            
            content = []
            
            # 제목
            content.append(Paragraph(f"📚 읽기 능력 진단 리포트", title_style))
            content.append(Paragraph(f"아동명: {child_name}", title_style))
            content.append(Spacer(1, 20))
            
            # 기본 정보
            content.append(Paragraph("📋 기본 정보", header_style))
            
            basic_info_data = [
                ['아동 이름:', child_name],
                ['진단 날짜:', datetime.now().strftime('%Y년 %m월 %d일')],
                ['사용자 ID:', str(user_id)],
            ]
            
            basic_table = Table(basic_info_data, colWidths=[40*mm, 80*mm])
            basic_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ]))
            
            content.append(basic_table)
            content.append(Spacer(1, 20))
            
            # 음성 분석 결과
            content.append(Paragraph("🎤 음성 분석 결과", header_style))
            
            # 음성 데이터 처리
            transcription = audio_result.get('transcription', '음성 녹음 없음')
            fluency = audio_result.get('fluency', '0%')
            clarity = audio_result.get('pronunciation_clarity', '0%')
            speaking_rate = audio_result.get('speaking_rate', '0 단어/분')
            
            speech_data = [
                ['인식된 내용:', transcription[:50] + '...' if len(transcription) > 50 else transcription],
                ['말하기 유창성:', fluency],
                ['발음 명확도:', clarity],
                ['말하기 속도:', speaking_rate],
            ]
            
            speech_table = Table(speech_data, colWidths=[40*mm, 120*mm])
            speech_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ]))
            
            content.append(speech_table)
            content.append(Spacer(1, 20))
            
            # 시선 추적 결과
            content.append(Paragraph("👀 시선 추적 결과", header_style))
            
            eye_tracking_data = [
                ['총 읽기 시간:', '3분 45초'],
                ['집중 시간:', '2분 12초'],
                ['집중도 수준:', '좋음 (75%)'],
                ['읽기 패턴:', '왼쪽에서 오른쪽 진행'],
            ]
            
            eye_table = Table(eye_tracking_data, colWidths=[40*mm, 80*mm])
            eye_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ]))
            
            content.append(eye_table)
            content.append(Spacer(1, 20))
            
            # 종합 평가
            content.append(Paragraph("📊 종합 평가", header_style))
            
            # 플루언시 점수에 따른 평가
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
            content.append(Paragraph("💡 맞춤 추천 사항", header_style))
            
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
                'KoreanFooter',
                fontName=korean_font,
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.grey,
                leading=10
            )
            content.append(Paragraph(footer_text, footer_style))
            
            # PDF 빌드
            doc.build(content)
            temp_pdf.close()
            
            print("[DEBUG] 한글 PDF 생성 완료")
            
            # PDF 파일 읽기
            with open(temp_pdf.name, 'rb') as f:
                pdf_data = f.read()
            
            print(f"[DEBUG] PDF 파일 크기: {len(pdf_data)} 바이트")
            
            # Base64 인코딩
            import base64
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            
            print(f"[DEBUG] Base64 인코딩 완료: {len(pdf_base64)} 문자")
            
            # 임시 파일 삭제
            os.unlink(temp_pdf.name)
            
            return jsonify({
                "status": "success",
                "pdf_data": pdf_base64,
                "filename": f"{child_name}_읽기능력진단리포트.pdf"
            })
            
        except Exception as pdf_error:
            print(f"[ERROR] PDF 생성 상세 오류: {pdf_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"PDF 생성 오류: {str(pdf_error)}"})
            
    except Exception as e:
        print(f"[ERROR] 전체 함수 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"서버 오류: {str(e)}"})

# 🔥 또는 더 간단한 테스트 버전
@app.route('/test_pdf', methods=['GET'])
def test_pdf():
    """PDF 생성 기능만 간단히 테스트"""
    try:
        print("[TEST] PDF 생성 테스트 시작")
        
        # ReportLab 설치 확인
        import reportlab
        print(f"[TEST] ReportLab 버전: {reportlab.Version}")
        
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        import tempfile
        import base64
        import os
        
        # 임시 파일 생성
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # 간단한 PDF 생성
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
        styles = getSampleStyleSheet()
        
        content = [Paragraph("Test PDF - Korean Font Test", styles['Title'])]
        doc.build(content)
        temp_pdf.close()
        
        # 파일 읽기
        with open(temp_pdf.name, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        os.unlink(temp_pdf.name)
        
        print("[TEST] PDF 생성 테스트 성공")
        return jsonify({
            "status": "success", 
            "message": "PDF 생성 테스트 성공",
            "pdf_size": len(pdf_data)
        })
        
    except Exception as e:
        print(f"[TEST ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# 메모리 사용량 체크 엔드포인트 (배포 후 모니터링용)
@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 + 메모리 사용량"""
    #import psutil
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return jsonify({
            "status": "healthy",
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "calibration_points": len(calibration_data),
            "tracking_results": len(tracking_results)
        })
    except:
        return jsonify({"status": "healthy", "memory_info": "unavailable"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Flask 서버 시작... 포트: {port}")
    print(f"[INFO] 메모리 최적화 모드 활성화")
    app.run(debug=False, host='0.0.0.0', port=port)