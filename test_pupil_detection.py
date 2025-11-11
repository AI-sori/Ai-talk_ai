# test_pupil_detection.py 새로 만들기

import cv2
import numpy as np

def test_webcam():
    """웹캠이 제대로 작동하는지 확인"""
    cap = cv2.VideoCapture(0)
    
    print("웹캠 테스트 시작 (q 누르면 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        
        cv2.imshow('Webcam Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("웹캠 테스트 완료")

def test_eye_detection():
    """MediaPipe로 눈 검출 테스트"""
    import mediapipe as mp
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    print("얼굴 검출 테스트 (q 누르면 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            print("✅ 얼굴 검출됨!")
            # 눈 주변에 점 그리기
            h, w, _ = frame.shape
            for landmark in results.multi_face_landmarks[0].landmark[:10]:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        else:
            print("❌ 얼굴 없음")
        
        cv2.imshow('Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== 테스트 시작 ===")
    print("1. 웹캠 테스트")
    test_webcam()
    
    print("\n2. 얼굴/눈 검출 테스트")
    test_eye_detection()
    
    print("\n=== 테스트 완료 ===")