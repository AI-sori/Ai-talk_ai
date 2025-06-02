import time
import numpy as np
import cv2

from gaze_tracker import GazeTracker

def run_gaze_diagnostics():
    print("[INFO] Gaze Diagnostics 시작")
    tracker = GazeTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    # ✅ 1. 보정 포인트 수동 수집 (실시간 영상 표시 포함)
    points = []
    for i in range(4):
        print(f"[CALIBRATION] 화면의 다른 위치를 바라보고 Enter 누르세요 ({i+1}/4)")
        collected = False
        while not collected:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture 실패")
                continue

            # 안내 메시지 시각적으로 표시
            cv2.putText(frame, f"Calibration {i+1}/4 - 응시 후 Enter", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                gaze = tracker.get_gaze_direction(frame)
                if gaze:
                    random_target = (np.random.randint(300, 1600), np.random.randint(300, 800))
                    print(f"[DEBUG] 보정용 gaze: {gaze}, target: {random_target}")
                    points.append({'gaze': gaze, 'target': random_target})
                    collected = True
                else:
                    print("[WARN] Gaze 추출 실패. 다시 시도하세요.")
            elif key == ord('q'):
                print("[INFO] 보정 중단됨.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cv2.destroyWindow("Calibration")

    if len(points) < 4:
        print("[ERROR] 보정 포인트 부족. 종료합니다.")
        return

    # ✅ 2. 보정 실행
    tracker.calibrate(points)

    print("[INFO] Starting gaze tracking diagnostics... Press 'q' to quit.")

    # ✅ 3. 실시간 시선 추적
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        result = tracker.track_reading(frame)

        if result:
            print(f"[TRACKED] Direction: {result['direction']} | "
                  f"Confidence: {result['confidence']:.2f} | "
                  f"Error(px): {result['error_offset']:.1f} | "
                  f"Pos: {result['position']}")
        else:
            print("[WARN] Tracking failed on this frame.")

        # 영상 위 시각화
        if result:
            cv2.putText(frame, f"{result['direction']} ({result['confidence']:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Gaze Diagnostics", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gaze_diagnostics()
