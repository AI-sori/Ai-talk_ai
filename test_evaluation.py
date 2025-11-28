# test_evaluation.py

import cv2
import time
from utils.gaze_tracker import GazeTracker

def test_gaze_evaluation():
    """시선 추적 평가 테스트"""
    
    print("=" * 60)
    print("시선 추적 정답 비율 테스트")
    print("=" * 60)
    print("10초 동안 화면을 왼쪽 → 중앙 → 오른쪽 순서로 보세요")
    print("=" * 60)
    
    # 트래커 초기화
    tracker = GazeTracker()
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다")
        return
    
    print("\n3초 후 시작...")
    time.sleep(3)
    
    # 평가 시작
    tracker.start_evaluation(duration=10.0)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 시선 추적
        result = tracker.track_reading(frame)
        
        # 화면 표시
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"Direction: {result['direction']}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        
        # 경과 시간
        if tracker.ground_truth_generator:
            elapsed = time.time() - tracker.ground_truth_generator.start_time
            cv2.putText(
                frame,
                f"Time: {elapsed:.1f}s / 10.0s",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )
            
            # 10초 경과 시 종료
            if elapsed >= 10.0:
                break
        
        cv2.imshow('Gaze Evaluation Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 결과 출력
    print("\n평가 완료! 결과 계산 중...")
    results = tracker.get_evaluation_results()
    
    # ✅ 디버깅 추가
    print(f"[DEBUG] 평가 모드: {tracker.evaluation_mode}")
    print(f"[DEBUG] 예측 개수: {len(tracker.evaluator.predictions)}")

    results = tracker.get_evaluation_results()
    if results:
        print("\n" + "=" * 60)
        print("✅ 최종 결과")
        print("=" * 60)
        print(f"정답 비율: {results['accuracy']:.2%}")
        print(f"정답 개수: {results['correct']} / {results['total']}")
        print(f"처리 프레임: {frame_count}개")
        print("=" * 60)
    else:
        print("[ERROR] 결과를 계산할 수 없습니다")


if __name__ == "__main__":
    test_gaze_evaluation()