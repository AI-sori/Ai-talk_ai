from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import time
import threading
import base64
from io import BytesIO
from PIL import Image

# 시선 추적 관련 모듈 임포트
from gaze_tracker import GazeTracker
from gaze_analyzer import GazeAnalyzer
from reading_diagnostics import ReadingDiagnostics
from text_manager import TextManager
from report_generator import PDFReportGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 변수 설정
camera = None
gaze_tracker = None
gaze_analyzer = None
reading_diagnostics = None
text_manager = None
current_state = "IDLE"
session_active = False
current_text_id = 0
processing_thread = None
thread_running = False
diagnostic_results = None
capture_frames = False
last_frame = None

# 카메라 및 모듈 초기화 함수
def initialize_system(model_path="models/best_resnet_model.pth", use_cuda=False):
    global camera, gaze_tracker, gaze_analyzer, reading_diagnostics, text_manager
    
    # 카메라 초기화
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        return False
    
    # 모델 파일 확인
    if not os.path.exists(model_path):
        print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 모듈 초기화
    gaze_tracker = GazeTracker(model_path, use_cuda=use_cuda)
    gaze_analyzer = GazeAnalyzer()
    reading_diagnostics = ReadingDiagnostics()
    text_manager = TextManager()
    
    print("시스템 초기화 완료")
    return True

# 시선 추적 및 처리 스레드
def process_frames():
    global camera, gaze_tracker, gaze_analyzer, current_state, last_frame, thread_running, capture_frames
    
    frame_counter=0
    thread_running = True
    while thread_running and capture_frames:
        ret, frame = camera.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 프레임 처리
        result_frame, gaze_vector = gaze_tracker.process_frame(frame)
        
        # 현재 상태가 읽기 또는 진단 중이면 시선 데이터 수집
        if current_state in ["READING", "DIAGNOSIS"] and gaze_vector is not None:
            # 픽셀 좌표 매핑
            x = int(frame.shape[1] / 2 - gaze_vector[0] * 300)
            y = int(frame.shape[0] / 2 + gaze_vector[1] * 300)
            
            # 시선 위치 추가
            gaze_analyzer.add_gaze_point(x, y)
            
            # 고정점 표시
            fixations = gaze_analyzer.detect_fixations()
            for _, _, fx, fy in fixations[-5:]:
                cv2.circle(result_frame, (int(fx), int(fy)), 5, (255, 0, 255), -1)
        
        # 마지막 프레임 저장
        last_frame = result_frame
        
        # 3프레임마다 한 번씩만 웹소켓으로 전송
        frame_counter += 1
        if frame_counter % 3 == 0:
            ret, buffer = cv2.imencode('.jpg', result_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})
        
        # 진단 모드이고 시간이 다 됐으면 진단 완료 처리
        if current_state == "DIAGNOSIS":
            global diagnostic_start_time, diagnostic_results
            elapsed = time.time() - diagnostic_start_time
            remaining = max(0, 5 - elapsed)
            
            if remaining <= 0:
                metrics = gaze_analyzer.calculate_reading_metrics()
                fixations = gaze_analyzer.fixations
                text_result = reading_diagnostics.end_text_session(fixations, metrics)
                
                print(f"텍스트 {text_manager.current_text_index + 1} 진단 완료!")
                
                # 다음 텍스트가 있으면 다음으로 이동, 없으면 리포트로
                if text_manager.has_next_text():
                    text_manager.move_to_next_text()
                    current_state = "TEXT_VIEW"
                    gaze_analyzer.reset()
                    socketio.emit('state_changed', {'state': current_state})
                else:
                    current_state = "REPORT"
                    diagnostic_results = text_result
                    socketio.emit('state_changed', {'state': current_state})
                    socketio.emit('diagnosis_complete', {'result': text_result})
        
        # 프레임 간 딜레이
        time.sleep(0.03)  # 약 30 FPS

# 라우트 설정
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report')
def report():
    return render_template('report.html')

# 웹소켓 이벤트 핸들러
@socketio.on('connect')
def handle_connect():
    print('클라이언트 연결됨')
    if not gaze_tracker:
        initialize_system()

@socketio.on('disconnect')
def handle_disconnect():
    print('클라이언트 연결 해제됨')
    global thread_running, capture_frames
    thread_running = False
    capture_frames = False

@socketio.on('start_camera')
def handle_start_camera():
    global capture_frames, processing_thread
    
    # 카메라 캡처 시작
    capture_frames = True
    
    # 이미 실행 중인 스레드가 없으면 새로 생성
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()
    
    emit('response', {'message': '카메라 시작됨'})

@socketio.on('stop_camera')
def handle_stop_camera():
    global capture_frames
    capture_frames = False
    emit('response', {'message': '카메라 중지됨'})

@socketio.on('change_state')
def handle_change_state(data):
    global current_state, current_text_id, session_active, diagnostic_start_time
    
    new_state = data.get('state')
    if new_state:
        current_state = new_state
        
        # 상태에 따른 추가 처리
        if current_state == "TEXT_VIEW":
            # 텍스트 정보 전송
            current_text = text_manager.get_current_text()
            if current_text:
                emit('text_data', {'text': current_text})
        
        elif current_state == "READING":
            # 읽기 세션 시작
            current_text = text_manager.get_current_text()
            if current_text:
                reading_diagnostics.start_text_session(
                    current_text['id'],
                    current_text['title']
                )
                gaze_analyzer.reset()
                session_active = True
        
        elif current_state == "DIAGNOSIS":
            # 진단 시작
            diagnostic_start_time = time.time()
        
        # 상태 변경 알림
        emit('state_changed', {'state': current_state})
        emit('response', {'message': f'상태 변경됨: {current_state}'})

@socketio.on('next_text')
def handle_next_text():
    global text_manager
    if text_manager.move_to_next_text():
        current_text = text_manager.get_current_text()
        emit('text_data', {'text': current_text})
        emit('response', {'message': f'텍스트 {text_manager.current_text_index + 1}로 이동'})

@socketio.on('prev_text')
def handle_prev_text():
    global text_manager
    if text_manager.move_to_prev_text():
        current_text = text_manager.get_current_text()
        emit('text_data', {'text': current_text})
        emit('response', {'message': f'텍스트 {text_manager.current_text_index + 1}로 이동'})

@socketio.on('generate_pdf')
def handle_generate_pdf():
    global reading_diagnostics
    
    # PDF 보고서 생성
    pdf_path = reading_diagnostics.save_pdf_report()
    if pdf_path:
        emit('pdf_generated', {'path': pdf_path})
        emit('response', {'message': f'PDF 보고서 생성됨: {pdf_path}'})
    else:
        emit('response', {'message': 'PDF 보고서 생성 실패'})

@app.route('/download_pdf/<path:filename>')
def download_pdf(filename):
    # 보안상의 이유로 filename 검증 필요
    if '..' in filename or filename.startswith('/'):
        return "잘못된 파일 경로", 400
    
    directory = os.path.abspath('reports')
    filepath = os.path.join(directory, os.path.basename(filename))
    
    if os.path.exists(filepath) and filepath.endswith('.pdf'):
        return send_file(filepath, as_attachment=True)
    else:
        return "파일을 찾을 수 없습니다", 404

# 메인 실행 블록
if __name__ == '__main__':
    initialize_system()
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)