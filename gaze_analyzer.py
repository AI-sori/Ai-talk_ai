# gaze_analyzer.py 생성
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import cv2

class GazeAnalyzer:
    def __init__(self, fixation_threshold=50, fixation_time=0.1):
        self.gaze_history = []  # (timestamp, x, y)
        self.current_fixation = None  # (start_time, points, center_x, center_y)
        self.fixations = []  # (start_time, duration, center_x, center_y)
        self.fixation_threshold = fixation_threshold  # 픽셀 단위
        self.fixation_time = fixation_time  # 초 단위
        self.diagnostic_system = None  # 진단 시스템 참조 추가
        self.reading_patterns = []  # 읽기 패턴 기록
        self.reading_metrics = {
            'saccade_count': 0,
            'regression_count': 0,
            'avg_fixation_duration': 0,
            'reading_speed': 0,
            'comprehension_score': 0
        }
    def add_gaze_point(self, x, y):
        """시선 위치 데이터 추가"""
        current_time = time.time()
        self.gaze_history.append((current_time, x, y))
        self._update_fixation(current_time, x, y)
        self._analyze_reading_pattern()

    def _update_fixation(self, timestamp, x, y):
        """시선 고정점 업데이트"""
        if self.current_fixation is None:
            # 새 고정점 시작
            self.current_fixation = (timestamp, [(x, y)], x, y)
            return
        
        # 기존 고정점과 새 위치 사이의 거리 계산
        start_time, points, center_x, center_y = self.current_fixation
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        if distance <= self.fixation_threshold:
            # 같은 고정점 내에 있음
            points.append((x, y))
            # 중심점 업데이트
            new_center_x = sum(p[0] for p in points) / len(points)
            new_center_y = sum(p[1] for p in points) / len(points)
            self.current_fixation = (start_time, points, new_center_x, new_center_y)
            
            # 고정점 시간이 임계값을 초과하면 완료된 고정으로 기록
            if timestamp - start_time >= self.fixation_time:
                duration = timestamp - start_time
                self.fixations.append((start_time, duration, new_center_x, new_center_y))
                self.current_fixation = None
        else:
            # 새로운 위치로 이동
            if len(points) > 3:  # 최소 포인트 수 확인
                duration = timestamp - start_time
                self.fixations.append((start_time, duration, center_x, center_y))
            
            # 새 고정점 시작
            self.current_fixation = (timestamp, [(x, y)], x, y)
    
        
    
    def _analyze_reading_pattern(self):
        """읽기 패턴 분석"""
        # 최소 2개의 고정점이 있어야 분석 가능
        if len(self.fixations) < 2:
            return
        
        # 가장 최근 두 고정점 가져오기
        prev_fixation = self.fixations[-2]
        curr_fixation = self.fixations[-1]
        
        # 수평 이동 분석 (왼쪽에서 오른쪽 = 정방향, 오른쪽에서 왼쪽 = 역행)
        prev_x = prev_fixation[2]
        curr_x = curr_fixation[2]
        
        if curr_x < prev_x - 20:  # 20픽셀 이상 왼쪽으로 이동 = 역행
            self.reading_patterns.append("regression")
            self.reading_metrics['regression_count'] += 1
        elif curr_x > prev_x + 20:  # 20픽셀 이상 오른쪽으로 이동 = 정방향
            self.reading_patterns.append("forward")
            self.reading_metrics['saccade_count'] += 1
        elif abs(curr_x - prev_x) <= 20:  # 수평 이동이 적음 = 같은 단어 재고정
            self.reading_patterns.append("refixation")
    
    def detect_fixations(self):
        """시선 고정점 감지"""
        return self.fixations[-20:]  # 최근 20개 고정점만 반환
    
    def generate_heatmap(self, width, height, sigma=30):
        """시선 위치에 따른 히트맵 생성"""
        if len(self.gaze_history) < 10:
            return None
        
        # 히트맵 생성
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 최근 100개 시선 포인트만 사용
        recent_points = [(x, y) for _, x, y in self.gaze_history[-100:]]
        
        # 가우시안 블러 적용
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # 각 포인트에 가우시안 블러 적용
        for x, y in recent_points:
            if 0 <= x < width and 0 <= y < height:
                # 단일 포인트 마스크 생성
                point_mask = np.zeros((height, width), dtype=np.uint8)
                point_mask[int(y), int(x)] = 255
                
                # 가우시안 블러 적용
                point_mask = cv2.GaussianBlur(point_mask, (51, 51), 0)
                
                # 히트맵에 추가
                intensity = point_mask / point_mask.max() * 255
                heatmap_point = np.zeros((height, width, 3), dtype=np.uint8)
                heatmap_point[:,:,0] = intensity  # 청록색 히트맵 (색상 조절 가능)
                heatmap = cv2.add(heatmap, heatmap_point)
        
        return heatmap
    
    def calculate_reading_metrics(self):
        """읽기 지표 계산"""
        if not self.fixations:
            return self.reading_metrics
        
        # 평균 고정 시간 계산
        durations = [duration for _, duration, _, _ in self.fixations]
        if durations:
            self.reading_metrics['avg_fixation_duration'] = sum(durations) / len(durations)
        
        # 읽기 속도 (고정점 수 / 전체 시간)
        if self.fixations:
            total_time = self.fixations[-1][0] - self.fixations[0][0]
            if total_time > 0:
                self.reading_metrics['reading_speed'] = len(self.fixations) / total_time
        
        return self.reading_metrics
    
    def analyze_reading_pattern(self):
        """읽기 패턴 분석 결과"""
        metrics = self.calculate_reading_metrics()
        
        # 역행 비율 (역행 수 / 전체 도약 수)
        regression_ratio = 0
        if metrics['saccade_count'] + metrics['regression_count'] > 0:
            regression_ratio = metrics['regression_count'] / (metrics['saccade_count'] + metrics['regression_count'])
        
        results = {
            'metrics': metrics,
            'regression_ratio': regression_ratio,
            'reading_pattern': 'normal'
        }
        
        # 읽기 패턴 분류
        if regression_ratio > 0.3:  # 30% 이상 역행이면 어려움이 있음
            results['reading_pattern'] = 'difficulty'
        elif metrics['avg_fixation_duration'] > 0.3:  # 평균 고정 시간이 300ms 이상이면 느린 읽기
            results['reading_pattern'] = 'slow'
        
        return results
    
    def reset(self):
        """데이터 초기화"""
        self.gaze_history = []
        self.current_fixation = None
        self.fixations = []
        self.reading_patterns = []
        self.reading_metrics = {
            'saccade_count': 0,
            'regression_count': 0,
            'avg_fixation_duration': 0,
            'reading_speed': 0,
            'comprehension_score': 0
        }
    def set_diagnostic_system(self, diagnostic_system):
        """진단 시스템 참조 설정"""
        self.diagnostic_system = diagnostic_system