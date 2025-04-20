# text_mapper.py 생성
import numpy as np
import cv2

class TextMapper:
    def __init__(self):
        self.text_regions = []  # (텍스트, 영역) 튜플 리스트
    
    def add_text_region(self, text, x1, y1, x2, y2):
        """텍스트 영역 추가"""
        self.text_regions.append((text, (x1, y1, x2, y2)))
    
    def clear_regions(self):
        """모든 텍스트 영역 삭제"""
        self.text_regions = []
    
    def create_line_regions(self, lines, start_y, line_height, margin_x=50):
        """여러 줄의 텍스트 영역 자동 생성"""
        screen_width = 800  # 기본값, 실제 화면 크기에 맞게 조정
        
        self.clear_regions()
        y = start_y
        
        for line in lines:
            # 한 줄 텍스트의 영역 설정
            self.add_text_region(
                line,
                margin_x, y, 
                screen_width - margin_x, y + line_height
            )
            y += line_height
    
    def map_gaze_to_text(self, gaze_screen_pos):
        """시선 위치를 기반으로 현재 읽고 있는 텍스트 반환"""
        x, y = gaze_screen_pos
        
        for text, (x1, y1, x2, y2) in self.text_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 추가적으로 단어 레벨 매핑도 가능
                return text, (x1, y1, x2, y2)
        
        return None, None

    def draw_text_regions(self, frame):
        """텍스트 영역 시각화"""
        for text, (x1, y1, x2, y2) in self.text_regions:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, text[:10] + "...", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame