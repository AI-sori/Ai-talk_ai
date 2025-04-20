# text_manager.py
class TextManager:
    def __init__(self):
        # 샘플 텍스트 데이터 (실제로는 파일에서 로드하거나 외부에서 가져옴)
        self.texts = [
            {
                'id': 1,
                'title': '1',
                'content': "뽀로로가 깡충깡충 뛰어갔어요! 빨리 도망가자! 뛰어, 뛰어, 깡충!"
            },
            {
                'id': 2,
                'title': '2',
                'content': '작은 토끼는 숲 속을 뛰어다니며, ‘퐁퐁퐁!’ 소리처럼 뛰었어요.'
            },
            {
                'id': 3,
                'title': '3',
                'content': '한 마리 작은 고양이는 자꾸만 눈을 깜빡였어요. 깜빡, 깜빡!'}
        ]
        self.current_text_index = 0
    
    def get_current_text(self):
        """현재 텍스트 가져오기"""
        if 0 <= self.current_text_index < len(self.texts):
            return self.texts[self.current_text_index]
        return None
    
    def move_to_next_text(self):
        """다음 텍스트로 이동"""
        if self.current_text_index < len(self.texts) - 1:
            self.current_text_index += 1
            return True
        return False
    
    def move_to_prev_text(self):
        """이전 텍스트로 이동"""
        if self.current_text_index > 0:
            self.current_text_index -= 1
            return True
        return False
    
    def reset(self):
        """처음 텍스트로 리셋"""
        self.current_text_index = 0
    
    def has_next_text(self):
        """다음 텍스트가 있는지 확인"""
        return self.current_text_index < len(self.texts) - 1
    
    def get_text_count(self):
        """총 텍스트 수 반환"""
        return len(self.texts)