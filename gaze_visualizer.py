import cv2
import numpy as np
import math
from utils.korean_text import put_korean_text

class GazeVisualizer:
    def __init__(self):
        pass
    
    def draw_gaze_direction(self, frame, eye_center, gaze_vector, length=100, color=(0, 0, 255), thickness=2):
        # 3D 벡터를 2D 평면에 투영
        x, y, z = gaze_vector
        
        # x와 y 방향 반전 (카메라가 거울처럼 보이는 것을 보정)
        x = -x*1.2
        y=-y*1.2
        
        # z 방향(깊이)을 고려하여 크기 조정
        # z가 음수일 때(카메라 방향으로 바라볼 때) 화살표는 더 짧아짐
        scale =  length *1.5
        
        # 중심점에서 벡터 방향으로 선 그리기
        end_point = (
            int(eye_center[0] + x * scale),
            int(eye_center[1] + y * scale)
        )
        
        cv2.arrowedLine(frame, eye_center, end_point, color, thickness)
        
        return frame
    
    def draw_eyes_bb(self, frame, left_eye_rect, right_eye_rect, color=(0, 255, 0), thickness=1):
        # 눈 사각형 그리기
        cv2.rectangle(
            frame, 
            (left_eye_rect[0], left_eye_rect[1]), 
            (left_eye_rect[2], left_eye_rect[3]), 
            color, 
            thickness
        )
        
        cv2.rectangle(
            frame, 
            (right_eye_rect[0], right_eye_rect[1]), 
            (right_eye_rect[2], right_eye_rect[3]), 
            color, 
            thickness
        )
        
        return frame
    
    def annotate_frame(self, frame, gaze_vector):
        # 시선 방향 텍스트 표시
        direction_text = self.get_gaze_direction(gaze_vector)
        
        cv2.putText(
            frame, 
            f"Gaze: {direction_text}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        # 벡터 값 직접 표시 (디버깅용)
        x, y, z = gaze_vector
        cv2.putText(
            frame, 
            f"Vector: x={x:.2f}, y={y:.2f}, z={z:.2f}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        return frame
    
    def get_gaze_direction(self, gaze_vector):
        # 3D 시선 벡터를 사람이 이해할 수 있는 방향으로 변환
        x, y, _ = gaze_vector  # z는 사용하지 않음
        x=-x *1.2#좌/우 반전
        y=-y *1.2#상/하 반전
        
        # 수평(좌우) 방향
        if x < -0.12:
            h_direction = "Right"
        elif x > 0.12:
            h_direction = "Left"
        else:
            h_direction = "Center"
        
        # 수직(상하) 방향
        if abs(y) < 0.12:  # 상하에 대한 더 넓은 범위
            v_direction = ""
        elif y > 0.12:
            v_direction = "Down"
        else:
            v_direction = "Up"
        
        return f"{v_direction} {h_direction}"
    