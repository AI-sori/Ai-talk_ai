# utils/evaluation.py

import time
import numpy as np

class ReadingGroundTruth:
    """읽기 시뮬레이션 정답 생성기"""
    
    def __init__(self, total_duration=10.0):
        """
        Args:
            total_duration: 읽기 총 시간 (초)
        """
        self.total_duration = total_duration
        self.start_time = None
    
    def start(self):
        """읽기 시작"""
        self.start_time = time.time()
        print(f"[INFO] Ground Truth 기록 시작 (총 {self.total_duration}초)")
    
    def get_true_direction(self, current_time=None):
        """
        현재 시간의 정답 방향 반환
        
        읽기 진행:
        - 0% ~ 33%: left (왼쪽에서 시작)
        - 33% ~ 67%: center (중간 읽는 중)
        - 67% ~ 100%: right (오른쪽 끝)
        
        Returns: 'left', 'center', 'right'
        """
        if self.start_time is None:
            return None
        
        if current_time is None:
            current_time = time.time()
        
        elapsed = current_time - self.start_time
        progress = elapsed / self.total_duration
        
        # 범위 초과 처리
        if progress > 1.0:
            progress = 1.0
        
        # 진행도에 따른 방향
        if progress < 0.33:
            return 'left'
        elif progress < 0.67:
            return 'center'
        else:
            return 'right'


class GazeAccuracyEvaluator:
    """시선 추적 정확도 평가"""
    
    def __init__(self):
        self.predictions = []  # 예측값
        self.ground_truth = []  # 정답
    
    def add_prediction(self, timestamp, predicted_direction):
        """예측 결과 추가"""
        self.predictions.append({
            'time': timestamp,
            'direction': predicted_direction
        })
    
    def set_ground_truth(self, ground_truth_list):
        """정답 설정"""
        self.ground_truth = ground_truth_list
    
    def calculate_accuracy(self):
        """
        정답 비율 계산
        
        Returns: {
            'accuracy': 0.0~1.0,
            'total': int,
            'correct': int
        }
        """
        if not self.predictions or not self.ground_truth:
            return None
        
        correct = 0
        total = 0
        
        for pred in self.predictions:
            # 가장 가까운 시간의 정답 찾기
            closest_gt = min(
                self.ground_truth,
                key=lambda x: abs(x['time'] - pred['time'])
            )
            
            pred_dir = pred['direction']
            true_dir = closest_gt['direction']
            
            if pred_dir == true_dir:
                correct += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


# 테스트 코드
if __name__ == "__main__":
    import time
    import random
    
    print("\n=== 평가 시스템 테스트 ===\n")
    
    # Ground Truth 생성
    gt_generator = ReadingGroundTruth(total_duration=10.0)
    gt_generator.start()
    
    # 평가기 생성
    evaluator = GazeAccuracyEvaluator()
    
    # 10초 동안 시뮬레이션
    print("10초 시뮬레이션 시작...")
    
    frame_times = []
    
    for i in range(100):  # 100 프레임
        time.sleep(0.1)
        
        current_time = time.time()
        frame_times.append(current_time)
        
        # 정답
        true_dir = gt_generator.get_true_direction(current_time)
        
        # 예측 (80% 정확도로 시뮬레이션)
        if random.random() < 0.8:
            pred_dir = true_dir
        else:
            pred_dir = random.choice(['left', 'center', 'right'])
        
        evaluator.add_prediction(current_time, pred_dir)
        
        if i % 10 == 0:
            elapsed = current_time - gt_generator.start_time
            print(f"{elapsed:.1f}초: 정답={true_dir}, 예측={pred_dir}")
    
    # Ground Truth 생성
    ground_truth_list = []
    for t in frame_times:
        direction = gt_generator.get_true_direction(t)
        ground_truth_list.append({'time': t, 'direction': direction})
    
    evaluator.set_ground_truth(ground_truth_list)
    
    # 결과
    results = evaluator.calculate_accuracy()
    
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"정답 비율: {results['accuracy']:.2%}")
    print(f"정답 개수: {results['correct']} / {results['total']}")
    print("=" * 60)