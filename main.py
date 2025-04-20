# main.py
import cv2
import numpy as np
import argparse
import os
import time
from gaze_tracker import GazeTracker
from gaze_analyzer import GazeAnalyzer
from reading_diagnostics import ReadingDiagnostics
from text_manager import TextManager
from utils.korean_text import put_korean_text


def main():
    parser = argparse.ArgumentParser(description="텍스트 읽기 추적 시스템")
    parser.add_argument("--model", default="models/best_resnet_model.pth", help="모델 파일 경로")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 번호")
    parser.add_argument("--cpu", action="store_true", help="CPU 모드 사용")
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model):
        print(f"경고: 모델 파일을 찾을 수 없습니다: {args.model}")
    
    # 시선 추적기 초기화 (모듈화된 코드 사용)
    gaze_tracker = GazeTracker(args.model, use_cuda=not args.cpu)
     
    # 시선 분석기 초기화 (새로 추가)
    gaze_analyzer = GazeAnalyzer()
    
    # 읽기 진단 초기화 (새로 추가)
    reading_diagnostics = ReadingDiagnostics()

    text_manager = TextManager()

    ##gaze_analyzer.set_diagnostic_system(reading_diagnostics)


    # 실제 시선 좌표 (캘리브레이션 후 얻은 값)-> 정확도 파악
    actual_gaze_points = np.array([])  # 초기에는 빈 배열로 시작, 캘리브레이션 중에 값이 채워짐
    
    # 카메라 열기
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 창 생성
    cv2.namedWindow("Reading Diagnostics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reading Diagnostics", 800, 600)
    
    print("시작하기 전에 캘리브레이션을 수행합니다.")
    # 처음에 캘리브레이션 수행
    if not gaze_tracker.calibrate_simple():
        print("캘리브레이션을 건너뛰었습니다. 정확도가 떨어질 수 있습니다.")
    
    # 세션 상태
    state = "IDLE" 
    session_active = False
    text_sessions_completed = []
    current_text = None
    diagnostic_mode = False
    diagnostic_start_time = 0
    diagnostic_results = None
    report_view = False
    
    # FPS 계산용 변수
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 프레임 처리 (모듈화된 코드 사용)
        result_frame, gaze_vector = gaze_tracker.process_frame(frame)
        
         # 시선 정확도 평가 (예측된 시선 좌표와 실제 시선 좌표 비교)
        if gaze_vector is not None and actual_gaze_points.size > 0:
            # 예측된 시선 좌표
            predicted_gaze_points = np.array([gaze_vector])  
            gaze_tracker.evaluate_accuracy(actual_gaze_points, predicted_gaze_points)

        
        # 화면 좌표로 변환
        if gaze_vector is not None and state in ["READING", "DIAGNOSIS"]:
            # 3D 벡터인 경우 x, y 값만 사용
            x_gaze = gaze_vector[0] if len(gaze_vector) >= 1 else 0
            y_gaze = gaze_vector[1] if len(gaze_vector) >= 2 else 0
            
            # 픽셀 좌표 매핑
            x = int(frame.shape[1] / 2 - x_gaze * 300)
            y = int(frame.shape[0] / 2 + y_gaze * 300)
            
            # 시선 위치 추가
            gaze_analyzer.add_gaze_point(x, y)
            
            # 고정점 표시
            fixations = gaze_analyzer.detect_fixations()
            for _, _, fx, fy in fixations[-5:]:  # 최근 5개만 표시
                cv2.circle(result_frame, (int(fx), int(fy)), 5, (255, 0, 255), -1)
        #상태에 따른 화면 표시
        # 수정할 부분 - "IDLE" 상태
        if state == "IDLE":
            # 시작 화면 - 한글 텍스트 사용
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 0, 0), 128)
            cv2.addWeighted(overlay, 0.5, result_frame, 0.5, 0, result_frame)
            
            # 한글 텍스트 렌더링
            result_frame = put_korean_text(
                result_frame, "읽기 능력 진단 시스템", 
                (frame.shape[1]//2 - 150, frame.shape[0]//2 - 30), 
                font_size=28, color=(0, 255, 255)
            )
            
            result_frame = put_korean_text(
                result_frame, "S: 진단 시작, Q: 종료", 
                (frame.shape[1]//2 - 120, frame.shape[0]//2 + 30), 
                font_size=24, color=(0, 255, 0)
            )

        # 수정할 부분 - "TEXT_VIEW" 상태
        elif state == "TEXT_VIEW":
            # 텍스트 보기 화면
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (30, 30), (frame.shape[1]-30, frame.shape[0]-30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
            
            # 한글 텍스트 렌더링
            result_frame = put_korean_text(
                result_frame, "텍스트 보기", 
                (50, 50), 
                font_size=28, color=(255, 255, 0)
            )
            
            current_text = text_manager.get_current_text()
            if current_text:
                text_title = f"텍스트 {text_manager.current_text_index + 1}/{text_manager.get_text_count()}: {current_text['title']}"
                result_frame = put_korean_text(
                    result_frame, text_title, 
                    (50, 100), 
                    font_size=24, color=(255, 255, 0)
                )
                
                # 텍스트 내용 표시
                lines = [current_text['content'][i:i+40] for i in range(0, len(current_text['content']), 40)]
                for i, line in enumerate(lines[:8]):  # 최대 8줄까지만 표시
                    y_pos = 150 + i * 30
                    result_frame = put_korean_text(
                        result_frame, line, 
                        (50, y_pos), 
                        font_size=20, color=(255, 255, 255)
                    )
            
            # 안내 메시지
            result_frame = put_korean_text(
                result_frame, "R: 읽기 시작, N: 다음, P: 이전, Q: 종료", 
                (50, frame.shape[0] - 50), 
                font_size=20, color=(0, 255, 0)
            )

        # 수정할 부분 - "READING" 상태
        elif state == "READING":
            # 읽기 화면
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (30, 30), (frame.shape[1]-30, frame.shape[0]-30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
            
            current_text = text_manager.get_current_text()
            if current_text:
                # 텍스트 제목과 내용 표시
                text_title = f"읽기: {current_text['title']}"
                result_frame = put_korean_text(
                    result_frame, text_title, 
                    (50, 50), 
                    font_size=24, color=(255, 255, 0)
                )
                
                # 텍스트 내용 표시
                lines = [current_text['content'][i:i+40] for i in range(0, len(current_text['content']), 40)]
                for i, line in enumerate(lines[:8]):
                    y_pos = 100 + i * 30
                    result_frame = put_korean_text(
                        result_frame, line, 
                        (50, y_pos), 
                        font_size=20, color=(255, 255, 255)
                    )
            
            # 안내 메시지
            result_frame = put_korean_text(
                result_frame, "D: 진단 완료, Q: 취소", 
                (50, frame.shape[0] - 50), 
                font_size=20, color=(0, 255, 0)
            )
            
            # 읽기 시간 표시
            reading_time = time.time() - reading_diagnostics.current_session['start_time']
            minutes = int(reading_time // 60)
            seconds = int(reading_time % 60)
            time_text = f"읽기 시간: {minutes}분 {seconds}초"
            result_frame = put_korean_text(
                result_frame, time_text, 
                (frame.shape[1] - 250, 50), 
                font_size=20, color=(0, 255, 255)
            )

        # 수정할 부분 - "DIAGNOSIS" 상태
        elif state == "DIAGNOSIS":
            # 진단 화면
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (30, 30), (frame.shape[1]-30, frame.shape[0]-30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
            
            # 카운트다운 시간
            elapsed = time.time() - diagnostic_start_time
            remaining = max(0, 5 - elapsed)
            
            # 카운트다운 표시
            result_frame = put_korean_text(
                result_frame, f"진단 중... {int(remaining)}초", 
                (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                font_size=28, color=(0, 0, 255)
            )
            
            # 진단 완료 처리는 기존 코드와 동일하게 유지
            # 진단 완료 처리
            if remaining <= 0:
                metrics = gaze_analyzer.calculate_reading_metrics()
                fixations = gaze_analyzer.fixations
                text_result = reading_diagnostics.end_text_session(fixations, metrics)
                
                print(f"텍스트 {text_manager.current_text_index + 1} 진단 완료!")
                
                # 다음 텍스트가 있으면 다음으로 이동, 없으면 리포트로
                if text_manager.has_next_text():
                    text_manager.move_to_next_text()
                    state = "TEXT_VIEW"
                    gaze_analyzer.reset()
                else:
                    state = "REPORT"
                    diagnostic_results = text_result  # 마지막 텍스트 결과
                    
        elif state == "REPORT":
            # 간소화된 리포트 화면 - PDF 저장 안내만 표시
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (20, 20), (frame.shape[1]-20, frame.shape[0]-20), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.9, result_frame, 0.1, 0, result_frame)
            
            # 진단 완료 메시지
            result_frame = put_korean_text(
                result_frame, "진단이 완료되었습니다!", 
                (frame.shape[1]//2 - 150, frame.shape[0]//2 - 100), 
                font_size=30, color=(255, 255, 255)
            )
            
            # PDF 저장 안내
            result_frame = put_korean_text(
                result_frame, "P 키를 눌러 PDF 보고서를 저장하세요", 
                (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                font_size=24, color=(0, 255, 255)
            )
            
            # 안내 메시지
            result_frame = put_korean_text(
                result_frame, "Space: 종료, R: 다시 시작, P: PDF 저장", 
                (frame.shape[1]//2 - 180, frame.shape[0] - 30), 
                font_size=20, color=(255, 255, 255)
            )
            
            # 자동으로 PDF 생성 옵션 (선택 사항)
            if diagnostic_results and not hasattr(state, 'pdf_generated'):
                # PDF 보고서 자동 생성
                pdf_path = reading_diagnostics.save_pdf_report()
                if pdf_path:
                    result_frame = put_korean_text(
                        result_frame, f"PDF 보고서가 저장되었습니다: {os.path.basename(pdf_path)}", 
                        (frame.shape[1]//2 - 200, frame.shape[0]//2 + 50), 
                        font_size=20, color=(0, 255, 0)
                    )
                    # 중복 생성 방지
                    state = type('obj', (object,), {'pdf_generated': True, 'name': 'REPORT'})
        
        # FPS 계산
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # FPS 표시 (디버깅용)
        if state != "REPORT":
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 표시
        cv2.imshow("Reading Diagnostics", result_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1)
        if key == ord('q'):  # q: 종료
            break
            
        elif key == ord('s') and state == "IDLE":  # s: 진단 시작
            state = "TEXT_VIEW"
            text_manager.reset()
            reading_diagnostics = ReadingDiagnostics()  # 새 진단 초기화
            gaze_analyzer.set_diagnostic_system(reading_diagnostics)
            
        elif key == ord('r') and state == "TEXT_VIEW":  # r: 읽기 시작
            state = "READING"
            current_text = text_manager.get_current_text()
            if current_text:
                reading_diagnostics.start_text_session(
                    current_text['id'],
                    current_text['title']
                )
                gaze_analyzer.reset()
                print(f"텍스트 {text_manager.current_text_index + 1} 읽기 시작!")
                
        elif key == ord('n') and state == "TEXT_VIEW":  # n: 다음 텍스트
            if text_manager.move_to_next_text():
                print(f"텍스트 {text_manager.current_text_index + 1}로 이동")
                
        elif key == ord('p') and state == "TEXT_VIEW":  # p: 이전 텍스트
            if text_manager.move_to_prev_text():
                print(f"텍스트 {text_manager.current_text_index + 1}로 이동")
                
        elif key == ord('d') and state == "READING":  # d: 읽기 완료, 진단 시작
            state = "DIAGNOSIS"
            diagnostic_start_time = time.time()
            print("진단 시작...")
            
        elif key == ord('r') and state == "REPORT":  # r: 처음부터 다시
            state = "IDLE"
            text_manager.reset()
            gaze_analyzer.reset()

        # 키 입력 처리 부분에 추가
        elif key == ord('p') and state == "REPORT":  # p: PDF 저장
            # PDF 보고서 생성
            pdf_path = reading_diagnostics.save_pdf_report()
            if pdf_path:
                # 안내 메시지 표시
                info_overlay = result_frame.copy()
                cv2.rectangle(info_overlay, (50, frame.shape[0]//2-50), (frame.shape[1]-50, frame.shape[0]//2+50), (0, 0, 0), -1)
                info_frame = cv2.addWeighted(info_overlay, 0.9, result_frame, 0.1, 0)
                
                # 메시지 표시
                if hasattr(reading_diagnostics, 'put_korean_text'):
                    info_frame = reading_diagnostics.put_korean_text(
                        info_frame, f"PDF 보고서가 저장되었습니다: {pdf_path}", 
                        (100, frame.shape[0]//2), font_size=20, color=(255, 255, 255)
                    )
                else:
                    cv2.putText(info_frame, f"PDF report saved: {pdf_path}", 
                            (100, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 메시지 표시
                cv2.imshow("Reading Diagnostics", info_frame)
                cv2.waitKey(2000)  # 2초 동안 메시지 표시
            
        elif key == 32 and state == "REPORT":  # space: 종료
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()