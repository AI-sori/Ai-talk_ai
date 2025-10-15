import gradio as gr
import cv2
import numpy as np
import json
import base64
import pymysql
import tempfile
import os
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import urllib.request
import ssl

# 모듈 import
try:
    from utils.gaze_tracker import GazeTracker
    from utils.audio_analyzer import AudioAnalyzer
    print("[INFO] 모듈 로드 성공")
except ImportError as e:
    print(f"[ERROR] 모듈 로드 실패: {e}")
    # 더미 클래스 (기존 코드와 동일)
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

# DB 설정
DB_CONFIG = {
    'host': 'svc.sel4.cloudtype.app',
    'port': 30213,
    'user': 'root',
    'password': 'ai-talk',
    'database': 'ai-talk',
    'charset': 'utf8mb4'
}

def get_db_connection():
    try:
        connection = pymysql.connect(**DB_CONFIG, connect_timeout=10)
        return connection
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        return None

# 전역 변수
gaze_tracker = None
audio_analyzer = None
calibration_data = []
tracking_results = []

# 독서 텍스트
STORIES = [
    "🐰 토끼와 거북이가 달리기를 했어요. 토끼는 빨리 뛰어갔지만 중간에 잠을 잤어요. 💤 거북이는 천천히 걸어갔어요. 🐢 결국 거북이가 먼저 도착했어요! 🏆",
    "🦁 사자가 쥐를 잡았어요. 쥐가 살려달라고 빌었어요. 나중에 사자가 그물에 걸렸을 때 쥐가 와서 그물을 갉아 사자를 구해주었어요. 🐭",
    "🐜 개미가 여름 내내 열심히 일했어요. 베짱이는 노래만 불렀어요. 겨울이 오자 개미는 따뜻한 집에 있었고 베짱이는 춥고 배고팠어요. 🎵"
]

current_story_index = 0

# ===== 핵심 함수들 =====

def init_system():
    """시스템 초기화"""
    global gaze_tracker, audio_analyzer
    try:
        gaze_tracker = GazeTracker()
        audio_analyzer = AudioAnalyzer()
        return "✅ 시스템 초기화 완료!"
    except Exception as e:
        return f"❌ 초기화 실패: {str(e)}"

def calibrate_gaze(image):
    """시선 보정 (5포인트 자동)"""
    global calibration_data
    
    if image is None:
        return "❌ 이미지가 없습니다", None
    
    try:
        # 5개 보정 포인트 (화면 4개 코너 + 중앙)
        h, w = image.shape[:2]
        calibration_points = [
            (w//4, h//4), (3*w//4, h//4),  # 상단 좌우
            (w//2, h//2),  # 중앙
            (w//4, 3*h//4), (3*w//4, 3*h//4)  # 하단 좌우
        ]
        
        calibration_data = []
        for target in calibration_points:
            gaze_point = gaze_tracker.get_gaze_direction(image)
            if gaze_point:
                calibration_data.append({
                    'target': target,
                    'gaze': gaze_point
                })
        
        if len(calibration_data) >= 4:
            success = gaze_tracker.calibrate(calibration_data)
            if success:
                return f"✅ 보정 완료! ({len(calibration_data)}개 포인트)", image
        
        return f"⚠️ 보정 부족 ({len(calibration_data)}/5)", image
        
    except Exception as e:
        return f"❌ 보정 오류: {str(e)}", None

def track_reading_video(image):
    """실시간 시선 추적"""
    global tracking_results
    
    if image is None or gaze_tracker is None:
        return image, "대기 중...", "0%", "0px"
    
    try:
        result = gaze_tracker.track_reading(image)
        
        if result:
            tracking_results.append({
                'timestamp': datetime.now().isoformat(),
                'gaze_direction': result['direction'],
                'confidence': result['confidence'],
                'position': result['position']
            })
            
            # 시각적 피드백 (화면에 점 표시)
            img_copy = image.copy()
            pos = result['position']
            cv2.circle(img_copy, (int(pos[0]), int(pos[1])), 10, (0, 255, 0), -1)
            
            return img_copy, result['direction'], f"{result['confidence']*100:.1f}%", f"{result['error_offset']:.0f}px"
    
    except Exception as e:
        print(f"[ERROR] 추적 오류: {e}")
    
    return image, "center", "50%", "40px"

def analyze_audio_file(audio_file):
    """음성 분석"""
    if audio_file is None:
        return "❌ 음성 파일이 없습니다"
    
    try:
        # audio_file은 Gradio에서 파일 경로로 제공됨
        with open(audio_file, 'rb') as f:
            # AudioAnalyzer는 Flask file object를 기대하므로 변환 필요
            class MockFileStorage:
                def __init__(self, filepath):
                    self.filepath = filepath
                    self.filename = os.path.basename(filepath)
                    with open(filepath, 'rb') as f:
                        self.content_length = len(f.read())
                
                def save(self, path):
                    import shutil
                    shutil.copy(self.filepath, path)
            
            mock_file = MockFileStorage(audio_file)
            result = audio_analyzer.analyze(mock_file)
            
            output = f"""
🎤 음성 분석 결과:
━━━━━━━━━━━━━━━
📝 인식 텍스트: {result['transcription']}
⏱️ 녹음 시간: {result['duration']}
📊 단어 수: {result['word_count']}개
🗣️ 말하기 속도: {result['speaking_rate']}
🎯 발음 명확도: {result['pronunciation_clarity']}
💬 유창성: {result['fluency']}
📖 이해도: {result['comprehension']}
            """
            return output, result
    
    except Exception as e:
        return f"❌ 분석 실패: {str(e)}", {}

def generate_report(user_id, child_name, audio_result):
    """진단 리포트 생성"""
    global tracking_results
    
    try:
        # 시선추적 분석
        if tracking_results:
            total_time = len(tracking_results) * 0.5
            center_count = sum(1 for r in tracking_results if r['gaze_direction'] == 'center')
            concentration = (center_count / len(tracking_results) * 100)
        else:
            total_time = 0
            concentration = 0
        
        report = {
            "user_id": user_id,
            "child_name": child_name,
            "diagnosis_date": datetime.now().strftime("%Y-%m-%d"),
            "reading_time": f"{total_time:.1f}초",
            "concentration": f"{concentration:.1f}%",
            "audio_result": audio_result
        }
        
        report_text = f"""
📋 읽기 능력 진단 리포트
━━━━━━━━━━━━━━━━━━━━
👦 아동명: {child_name}
📅 진단일: {report['diagnosis_date']}
⏱️ 읽기 시간: {report['reading_time']}
🎯 집중도: {report['concentration']}

🎤 음성 분석:
  • 발음: {audio_result.get('pronunciation_clarity', 'N/A')}
  • 유창성: {audio_result.get('fluency', 'N/A')}
  • 이해도: {audio_result.get('comprehension', 'N/A')}

💡 추천사항:
  • 매일 20분 소리내어 읽기
  • 시선 집중력 훈련
  • 발음 교정 연습
        """
        
        return report_text, report
    
    except Exception as e:
        return f"❌ 리포트 생성 실패: {str(e)}", {}

def generate_pdf(report_data):
    """PDF 생성 및 DB 저장"""
    try:
        # 나눔고딕 폰트 다운로드
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            font_url = "https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.ttf"
            font_request = urllib.request.Request(font_url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(font_request, timeout=15, context=ssl_context) as response:
                font_data = response.read()
            
            temp_font = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
            temp_font.write(font_data)
            temp_font.close()
            
            pdfmetrics.registerFont(TTFont('NanumGothic', temp_font.name))
            font_name = 'NanumGothic'
        except:
            font_name = 'Helvetica'
        
        # PDF 생성
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4, topMargin=25*mm, bottomMargin=20*mm)
        
        content = []
        
        # 스타일
        title_style = ParagraphStyle('Title', fontName=font_name, fontSize=20, alignment=TA_CENTER, textColor=colors.HexColor('#2c3e50'))
        
        # 제목
        content.append(Paragraph("📚 읽기 능력 진단 리포트", title_style))
        content.append(Spacer(1, 20))
        
        # 내용 추가 (간소화)
        basic_data = [
            ['아동 이름', report_data['child_name']],
            ['진단 날짜', report_data['diagnosis_date']],
            ['집중도', report_data['concentration']]
        ]
        
        table = Table(basic_data, colWidths=[50*mm, 100*mm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ]))
        
        content.append(table)
        doc.build(content)
        temp_pdf.close()
        
        # PDF 파일 읽기
        with open(temp_pdf.name, 'rb') as f:
            pdf_data = f.read()
        
        # DB 저장
        connection = get_db_connection()
        if connection:
            with connection.cursor() as cursor:
                sql = "INSERT INTO pdf_reports (member_id, child_name, pdf_data, filename, created_at) VALUES (%s, %s, %s, %s, NOW())"
                cursor.execute(sql, (report_data['user_id'], report_data['child_name'], pdf_data, f"{report_data['child_name']}_리포트.pdf"))
                connection.commit()
            connection.close()
        
        # 임시 파일 정리
        os.unlink(temp_pdf.name)
        if 'temp_font' in locals():
            os.unlink(temp_font.name)
        
        return temp_pdf.name  # Gradio File 컴포넌트용
    
    except Exception as e:
        print(f"[ERROR] PDF 생성 실패: {e}")
        return None

# ===== Gradio UI =====

with gr.Blocks(title="읽기 능력 진단 시스템", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 AI 읽기 능력 진단 시스템")
    
    # 상태 변수
    audio_result_state = gr.State({})
    report_state = gr.State({})
    
    with gr.Tab("1️⃣ 시스템 초기화"):
        init_btn = gr.Button("🚀 시스템 초기화", variant="primary")
        init_output = gr.Textbox(label="상태")
        init_btn.click(init_system, outputs=init_output)
    
    with gr.Tab("2️⃣ 시선 보정"):
        gr.Markdown("### 카메라를 보며 화면 중앙을 응시하세요")
        calib_image = gr.Image(source="webcam", streaming=True)
        calib_btn = gr.Button("🎯 보정 시작", variant="primary")
        calib_output = gr.Textbox(label="보정 결과")
        calib_btn.click(calibrate_gaze, inputs=calib_image, outputs=[calib_output, calib_image])
    
    with gr.Tab("3️⃣ 독서 추적"):
        gr.Markdown(f"### 📖 다음 이야기를 읽어주세요:\n\n{STORIES[0]}")
        
        tracking_video = gr.Image(source="webcam", streaming=True)
        
        with gr.Row():
            direction_out = gr.Textbox(label="시선 방향", value="center")
            confidence_out = gr.Textbox(label="신뢰도", value="0%")
            error_out = gr.Textbox(label="오차", value="0px")
        
        tracking_video.stream(track_reading_video, inputs=tracking_video, outputs=[tracking_video, direction_out, confidence_out, error_out])
    
    with gr.Tab("4️⃣ 음성 분석"):
        gr.Markdown("### 🎤 읽은 내용에 대해 말씀해주세요")
        audio_input = gr.Audio(source="microphone", type="filepath", label="음성 녹음")
        analyze_btn = gr.Button("분석 시작", variant="primary")
        audio_output = gr.Textbox(label="분석 결과", lines=10)
        
        analyze_btn.click(analyze_audio_file, inputs=audio_input, outputs=[audio_output, audio_result_state])
    
    with gr.Tab("5️⃣ 리포트 생성"):
        gr.Markdown("### 📋 진단 정보 입력")
        user_id_input = gr.Number(label="사용자 ID", value=1)
        child_name_input = gr.Textbox(label="아동 이름", placeholder="이름을 입력하세요")
        
        generate_btn = gr.Button("📄 리포트 생성", variant="primary")
        report_output = gr.Textbox(label="리포트", lines=15)
        
        pdf_btn = gr.Button("💾 PDF 다운로드")
        pdf_output = gr.File(label="PDF 파일")
        
        generate_btn.click(
            generate_report,
            inputs=[user_id_input, child_name_input, audio_result_state],
            outputs=[report_output, report_state]
        )
        
        pdf_btn.click(generate_pdf, inputs=report_state, outputs=pdf_output)

# 실행
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)