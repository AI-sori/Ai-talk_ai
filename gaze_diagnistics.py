
import time
import numpy as np
import cv2

from gaze_tracker import GazeTracker

def run_gaze_diagnostics():
    tracker = GazeTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    print("[INFO] Starting gaze tracking diagnostics... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        start_time = time.time()
        result = tracker.track_reading(frame)
        elapsed = time.time() - start_time

        if result:
            print(f"[TRACKED] Direction: {result['direction']} | "
                  f"Confidence: {result['confidence']:.2f} | "
                  f"Error(px): {result['error_offset']:.1f} | "
                  f"Pos: {result['position']} | "
                  f"Time: {elapsed*1000:.1f} ms")
        else:
            print("[WARN] Tracking failed on this frame.")

        # 화면에도 표시 (선택 사항)
        cv2.putText(frame, f"Direction: {result['direction'] if result else 'N/A'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gaze Diagnostics", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gaze_diagnostics()
