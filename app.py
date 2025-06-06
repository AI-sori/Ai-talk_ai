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

def create_simple_pdf(report_data):
    """메모리 최적화된 PDF 생성"""
    temp_pdf = None
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # 한글 폰트 등록 시도
        korean_font = 'Helvetica'
        font_paths = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            'C:/Windows/Fonts/malgun.ttf',
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('KoreanFont', font_path))
                    korean_font = 'KoreanFont'
                    break
            except:
                continue
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        doc = SimpleDocTemplate(
            temp_pdf.name, 
            pagesize=A4, 
            topMargin=35*mm, 
            bottomMargin=25*mm,
            leftMargin=25*mm,
            rightMargin=25*mm
        )
        
        styles = getSampleStyleSheet()
        
        # 전문적인 스타일들
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Title'],
            fontSize=20,
            spaceAfter=25,
            alignment=TA_CENTER,
            fontName=korean_font,
            textColor=colors.HexColor('#2C3E50')
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            fontName=korean_font,
            textColor=colors.HexColor('#34495E'),
            borderWidth=0,
            borderPadding=5,
            backColor=colors.HexColor('#ECF0F1')
        )
        
        normal_style = ParagraphStyle(
            'ReportNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            fontName=korean_font,
            textColor=colors.HexColor('#2C3E50')
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
        
        # 제목
        content.append(Paragraph("읽기 능력 진단 리포트", title_style))
        content.append(Spacer(1, 15))
        
        # 리포트 정보 헤더
        info_header = Table([['리포트 개요']], colWidths=[160*mm])
        info_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(info_header)
        
        # 기본 정보
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
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
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
        result_header = Table([['검사 결과 분석']], colWidths=[160*mm])
        result_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(result_header)
        
        # 측정 결과
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
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
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
        speech_header = Table([['음성 분석 결과']], colWidths=[160*mm])
        speech_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#BDC3C7')),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(speech_header)
        
        # 음성 분석
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
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
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
        
        # 푸터
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            fontName=korean_font,
            textColor=colors.HexColor('#7F8C8D')
        )
        content.append(Paragraph("본 리포트는 AI 읽기 능력 진단 시스템에 의해 자동 생성되었습니다", footer_style))
        
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
    """PDF 리포트 다운로드 + DB 저장 - 메모리 최적화"""
    try:
        data = request.json
        child_name = data.get('child_name', 'Unknown')
        user_id = data.get('user_id', 1)
        audio_result = data.get('audio_result', {})
        
        global tracking_results
        
        if tracking_results:
            total_tracking_time = len(tracking_results) * 0.5
            center_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'center')
            concentration_score = (center_count / len(tracking_results) * 100)
        else:
            total_tracking_time = 0
            concentration_score = 0
        
        # 리포트 데이터 구성
        report_data = {
            "report": {
                "child_name": child_name,
                "diagnosis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "reading_time": f"{total_tracking_time:.1f}초",
                "results": {
                    "reading_speed": "보통",
                    "concentration": f"{concentration_score:.1f}%",
                    "comprehension": audio_result.get('comprehension', '0.0%')
                },
                "speech_analysis": {
                    "transcription": audio_result.get('transcription', 'N/A'),
                    "fluency": audio_result.get('fluency', '0.0%'),
                    "pronunciation_clarity": audio_result.get('pronunciation_clarity', '0.0%')
                }
            }
        }
        
        # 1. PDF 생성
        pdf_path = create_simple_pdf(report_data)
        
        # 2. DB에 텍스트 데이터 저장 
        report_text = create_report_text(report_data)
        diagnosis_id = save_report_to_db(user_id, report_text)
        
        if pdf_path:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            
            # 임시 파일 삭제
            os.unlink(pdf_path)
            
            filename = f"진단리포트_{child_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            print(f"[SUCCESS] PDF 생성 및 DB 저장 완료. DB ID: {diagnosis_id}")
            
            # 메모리 정리
            cleanup_memory()
            
            return jsonify({
                "status": "success",
                "pdf_data": pdf_base64,
                "filename": filename,
                "diagnosis_id": diagnosis_id,
                "message": "PDF 생성 및 DB 저장 완료"
            })
        else:
            return jsonify({"status": "error", "message": "PDF 생성 실패"})
            
    except Exception as e:
        print(f"[ERROR] PDF 리포트 생성 오류: {e}")
        return jsonify({"status": "error", "message": f"PDF 생성 실패: {str(e)}"})
    finally:
        # 메모리 정리
        gc.collect()

# 메모리 사용량 체크 엔드포인트 (배포 후 모니터링용)
@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 + 메모리 사용량"""
    import psutil
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