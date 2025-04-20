# reading_diagnostics.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import cv2
import time
import datetime

class ReadingDiagnostics:
    def __init__(self):
        self.reading_sessions = []  # 세션 기록
        self.text_sessions = []     # 텍스트별 세션 기록
        
        self.current_session = {
            'start_time': None,
            'end_time': None,
            'fixations': [],
            'metrics': {},
            'focus_periods': [],
            'child_name': '홍길동',  # 기본 이름
            'text_id': 0,           # 현재 텍스트 ID
            'text_title': ""        # 텍스트 제목
        }
        
        self.diagnostic_results = {
            'reading_speed': '적당함',
            'focus_level': '보통',
            'comprehension': '보통',
            'reading_pattern': '일반적인 읽기',
            'main_issue': '특별한 문제점이 발견되지 않음',
            'fluency_score': 70,
            'comprehension_level': '중',
            'focus_time': 0,
            'recommendations': ['규칙적인 독서 시간 – 매일 15분씩 책 읽는 습관 기르기'],
            'oneliner_feedback': '꾸준한 독서로 좋은 습관을 유지해보세요',
            'next_diagnosis_date': (datetime.datetime.now() + datetime.timedelta(days=14)).strftime('%Y.%m.%d')
        }
    
    def set_child_name(self, name):
        """아이 이름 설정"""
        self.current_session['child_name'] = name
    
    def start_text_session(self, text_id, text_title=""):
        """특정 텍스트에 대한 읽기 세션 시작"""
        self.current_session = {
            'start_time': time.time(),
            'end_time': None,
            'fixations': [],
            'metrics': {},
            'focus_periods': [],
            'child_name': self.current_session.get('child_name', '홍길동'),
            'text_id': text_id,
            'text_title': text_title
        }
        return True
    
    def start_session(self):
        """읽기 세션 시작 (호환성 유지)"""
        return self.start_text_session(0, "일반 세션")
    
    def record_focus_period(self, is_focused):
        """집중 상태 기록"""
        current_time = time.time()
        
        # 세션이 활성화된 상태일 때만 기록
        if self.current_session.get('start_time') is not None:
            self.current_session.setdefault('focus_periods', []).append((current_time, is_focused))
    
    def end_text_session(self, fixations, metrics):
        """특정 텍스트 세션 종료 및 결과 저장"""
        self.current_session['end_time'] = time.time()
        self.current_session['fixations'] = fixations
        self.current_session['metrics'] = metrics
        
        # 집중 시간 계산
        focus_time = self._calculate_focus_time()
        self.current_session['focus_time'] = focus_time
        
        # 개별 텍스트 진단 결과 계산
        text_result = self._calculate_text_results(focus_time)
        self.current_session['result'] = text_result
        
        # 텍스트 세션 기록에 추가
        if not hasattr(self, 'text_sessions'):
            self.text_sessions = []
        
        self.text_sessions.append(self.current_session.copy())
        
        # 읽기 기록에도 추가 (호환성 유지)
        self.reading_sessions.append(self.current_session.copy())
        
        return text_result
    
    def end_session(self, fixations, metrics):
        """세션 종료 (호환성 유지)"""
        return self.end_text_session(fixations, metrics)
    
    def _calculate_focus_time(self):
        """집중 시간 계산"""
        if not hasattr(self, 'current_session') or not self.current_session.get('focus_periods', []):
            # 집중도 데이터가 없으면 세션 시간의 70%를 집중했다고 가정
            session_duration = 0
            if self.current_session.get('start_time') is not None:
                end_time = self.current_session.get('end_time', time.time())
                session_duration = end_time - self.current_session['start_time']
            return session_duration * 0.7
        
        focus_time = 0
        prev_time = self.current_session.get('start_time', 0)
        prev_focused = False
        
        for time_point, is_focused in self.current_session.get('focus_periods', []):
            if prev_focused:
                focus_time += (time_point - prev_time)
            
            prev_time = time_point
            prev_focused = is_focused
        
        # 마지막 기록이 집중 상태였다면 세션 종료 시간까지 계산
        if prev_focused and self.current_session.get('end_time') is not None:
            focus_time += (self.current_session['end_time'] - prev_time)
        
        return focus_time
    
    def _calculate_text_results(self, focus_time):
        """개별 텍스트에 대한 진단 결과 계산"""
        metrics = self.current_session.get('metrics', {})
        start_time = self.current_session.get('start_time', time.time())
        end_time = self.current_session.get('end_time', time.time())
        session_duration = end_time - start_time
        
        # 기본값 설정
        reading_speed = metrics.get('reading_speed', 1.8)  # 기본 속도 (중간)
        saccade_count = metrics.get('saccade_count', 10)   # 기본 도약 수
        regression_count = metrics.get('regression_count', 3)  # 기본 역행 수
        avg_fixation_duration = metrics.get('avg_fixation_duration', 0.25)  # 기본 고정 시간
        
        # 유효한 값이 없으면 기본값으로 대체
        if saccade_count <= 0:
            saccade_count = 10
        if avg_fixation_duration <= 0:
            avg_fixation_duration = 0.25
        
        # 유창성 점수 계산 (0-100)
        regression_ratio = regression_count / max(1, saccade_count)
        fluency_score = 100 - (regression_ratio * 50) - (min(avg_fixation_duration, 0.5) * 100)
        fluency_score = max(0, min(100, fluency_score))
        
        # 이해도 수준 분류
        comprehension_level = '중'
        if fluency_score >= 80:
            comprehension_level = '상'
        elif fluency_score >= 50:
            comprehension_level = '중'
        else:
            comprehension_level = '하'
        
        # 읽기 패턴 분류
        reading_pattern = '일반적인 읽기'
        if regression_ratio > 0.3:
            reading_pattern = '어려움 있음 (잦은 역행)'
        elif avg_fixation_duration > 0.3:
            reading_pattern = '느린 읽기'
        
        # 주요 이슈 식별
        main_issue = self._identify_main_issue(metrics, focus_time, session_duration)
        
        # 진단 결과 업데이트 (마지막 세션 결과로 저장)
        self.diagnostic_results.update({
            'reading_speed': self._evaluate_reading_speed(reading_speed),
            'focus_level': self._evaluate_focus_level(focus_time, session_duration),
            'comprehension': self._evaluate_comprehension(metrics),
            'reading_pattern': reading_pattern,
            'main_issue': main_issue,
            'fluency_score': round(fluency_score),
            'comprehension_level': comprehension_level,
            'focus_time': focus_time
        })
        
        # 추천사항 생성
        recommendations = []
        if regression_ratio > 0.3:
            recommendations.append('역행이 많이 발생했습니다. 단어 인식력을 높이기 위한 훈련이 필요합니다.')
        if avg_fixation_duration > 0.3:
            recommendations.append('고정 시간이 길어 읽기 속도가 느립니다. 시각적 단어 인식 훈련이 도움이 될 수 있습니다.')
        if regression_count > 10:
            recommendations.append('읽기 중 되돌아가는 횟수가 많습니다. 텍스트 난이도를 낮추거나 친숙한 주제로 연습하세요.')
        if len(recommendations) == 0:
            recommendations.append('규칙적인 독서 시간 – 매일 15분씩 책 읽는 습관 기르기')
        
        self.diagnostic_results['recommendations'] = recommendations
        
        # 텍스트 세션 결과 반환
        return {
            'reading_speed': self._evaluate_reading_speed(reading_speed),
            'focus_level': self._evaluate_focus_level(focus_time, session_duration),
            'comprehension': self._evaluate_comprehension(metrics),
            'main_issue': main_issue,
            'session_duration': session_duration,
            'focus_time': focus_time,
            'fluency_score': round(fluency_score),
            'reading_pattern': reading_pattern,
            'comprehension_level': comprehension_level
        }
    
    def generate_scanpath_visualization(self, width, height):
        """시선 경로 시각화"""
        if not self.current_session.get('fixations', []):
            return None
        
        try:
            # Matplotlib 그림 생성
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 시선 경로 그리기
            fixations = self.current_session['fixations']
            x_coords = [fx for _, _, fx, _ in fixations]
            y_coords = [fy for _, _, _, fy in fixations]
            
            # 점 그리기
            ax.scatter(x_coords, y_coords, s=30, c=range(len(x_coords)), cmap='viridis', alpha=0.7)
            
            # 선 그리기
            for i in range(1, len(x_coords)):
                ax.plot([x_coords[i-1], x_coords[i]], [y_coords[i-1], y_coords[i]], 'r-', alpha=0.5)
            
            # 그래프 설정
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # y축 반전 (이미지 좌표계)
            ax.set_title('시선 경로 분석')
            ax.set_xlabel('X 좌표')
            ax.set_ylabel('Y 좌표')
            
            # 이미지로 변환
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # OpenCV로 읽기
            buf_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(buf_arr, cv2.IMREAD_COLOR)
            
            return img
        except Exception as e:
            print(f"시각화 생성 오류: {e}")
            return None
    
    def generate_combined_report(self):
            """한글로 통합 보고서 생성"""
            # 세션 데이터가 없는 경우
            if not self.text_sessions and not self.reading_sessions:
                return "읽기 분석 리포트\n\n세션 데이터가 없습니다."
            
            # 통합 결과 생성
            results = self.diagnostic_results
            
            # 시간 형식 변환 (초 -> 분:초)
            sessions = self.text_sessions if self.text_sessions else self.reading_sessions
            total_duration = sum(session.get('end_time', 0) - session.get('start_time', 0) for session in sessions)
            
            total_minutes = int(total_duration // 60)
            total_seconds = int(total_duration % 60)
            
            # 날짜 생성
            today = datetime.datetime.now().strftime('%Y.%m.%d')
            
            # 한글 보고서 생성
            report = f"""
        읽기 분석 리포트

        아이 이름: {self.current_session.get('child_name', '홍길동')}
        진단 날짜: {today}
        읽은 텍스트 수: {len(sessions)}개
        총 읽기 시간: {total_minutes}분 {total_seconds}초

        분석 결과:
        - 읽기 속도: {results.get('reading_speed', '보통')}
        - 집중력: {results.get('focus_level', '보통')}
        - 이해력: {results.get('comprehension', '보통')}

        시선 추적 결과:
        - 주요 문제: {results.get('main_issue', '특별한 문제점이 발견되지 않음')}

        추천 활동:
        """
            
            # 추천 사항 추가
            recommendations = results.get('recommendations', [])
            if not recommendations:
                recommendations = ["규칙적인 독서 시간 – 매일 15분씩 책 읽는 습관 기르기"]
            
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            
            return report

    
    def _most_common(self, items):
        """리스트에서 가장 빈번한 항목 찾기"""
        if not items:
            return None
        
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _evaluate_reading_speed(self, speed):
        """읽기 속도 평가"""
        if speed > 2.5:
            return '빠름'
        elif speed > 1.5:
            return '적당함'
        else:
            return '조금 느림'
    
    def _evaluate_focus_level(self, focus_time, session_duration):
        """집중력 평가"""
        focus_ratio = focus_time / max(session_duration, 0.001)
        if focus_ratio > 0.8:
            return '높음'
        elif focus_ratio > 0.6:
            return '보통'
        else:
            return '주의 필요'
    
    def _evaluate_comprehension(self, metrics):
        """이해력 평가"""
        saccade_count = metrics.get('saccade_count', 10)
        regression_count = metrics.get('regression_count', 3)
        
        # 유효한 값이 없으면 기본값으로 대체
        if saccade_count <= 0:
            saccade_count = 10
        
        regression_ratio = regression_count / max(saccade_count, 1)
        if regression_ratio < 0.15:
            return '매우 좋음'
        elif regression_ratio < 0.3:
            return '보통'
        else:
            return '더 연습이 필요해요'
    
    def _identify_main_issue(self, metrics, focus_time, session_duration):
        """주요 문제점 식별"""
        saccade_count = metrics.get('saccade_count', 10)
        regression_count = metrics.get('regression_count', 3)
        avg_fixation_duration = metrics.get('avg_fixation_duration', 0.25)
        
        # 유효한 값이 없으면 기본값으로 대체
        if saccade_count <= 0:
            saccade_count = 10
        
        regression_ratio = regression_count / max(saccade_count, 1)
        focus_ratio = focus_time / max(session_duration, 0.001)
        
        if regression_ratio > 0.3:
            return '같은 부분을 여러 번 읽는 패턴'
        elif avg_fixation_duration > 0.3:
            return '특정 단어에서 오래 멈추는 현상'
        elif focus_ratio < 0.6:
            return '시선이 자주 흔들리는 패턴'
        else:
            return '특별한 문제점이 발견되지 않음'
    
    def reset(self):
        """데이터 초기화"""
        self.current_session = {
            'start_time': None,
            'end_time': None,
            'fixations': [],
            'metrics': {},
            'focus_periods': [],
            'child_name': self.current_session.get('child_name', '홍길동'),
            'text_id': 0,
            'text_title': ""
        }
        self.diagnostic_results = {
            'reading_speed': '적당함',
            'focus_level': '보통',
            'comprehension': '보통',
            'reading_pattern': '일반적인 읽기',
            'main_issue': '특별한 문제점이 발견되지 않음',
            'fluency_score': 70,
            'comprehension_level': '중',
            'focus_time': 0,
            'recommendations': ['규칙적인 독서 시간 – 매일 15분씩 책 읽는 습관 기르기'],
            'oneliner_feedback': '꾸준한 독서로 좋은 습관을 유지해보세요',
            'next_diagnosis_date': (datetime.datetime.now() + datetime.timedelta(days=14)).strftime('%Y.%m.%d')
        }

    def save_pdf_report(self, output_path="reports"):
        """PDF 보고서 저장"""
        try:
            from report_generator import PDFReportGenerator
            
            # 보고서 생성기 초기화
            generator = PDFReportGenerator(output_path)
            
            # 현재 진단 결과와 고정점 데이터로 보고서 생성
            child_name = self.current_session.get('child_name', '홍길동')
            
            # 진단 데이터가 없으면 기본 진단 결과 사용
            diagnostic_data = self.diagnostic_results.copy()
            
            # 세션 시간 정보 추가
            if self.current_session.get('start_time') is not None and self.current_session.get('end_time') is not None:
                diagnostic_data['session_duration'] = self.current_session['end_time'] - self.current_session['start_time']
            
            # 고정점 데이터
            fixations = self.current_session.get('fixations', [])
            
            # PDF 생성
            pdf_path = generator.generate_pdf(diagnostic_data, fixations, child_name)
            
            # 생성된 PDF 파일 자동으로 열기
            import os
            import platform
            import subprocess
            
            # 파일과 폴더의 절대 경로 구하기
            abs_pdf_path = os.path.abspath(pdf_path)
            folder_path = os.path.dirname(abs_pdf_path)
            
            # 운영체제에 따라 파일 열기
            try:
                if platform.system() == 'Windows':
                    os.startfile(abs_pdf_path)  # PDF 파일 열기
                    # 파일 탐색기로 폴더 열기
                    subprocess.Popen(f'explorer "{folder_path}"')
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', abs_pdf_path])  # PDF 파일 열기
                    subprocess.call(['open', folder_path])   # 폴더 열기
                else:  # Linux
                    subprocess.call(['xdg-open', abs_pdf_path])  # PDF 파일 열기
                    subprocess.call(['xdg-open', folder_path])   # 폴더 열기
                
                print(f"PDF 파일이 생성되었습니다: {abs_pdf_path}")
                print(f"PDF 파일이 자동으로 열렸습니다.")
            except Exception as e:
                print(f"PDF 파일을 자동으로 열지 못했습니다: {e}")
            
            return pdf_path
        except ImportError:
            print("ReportLab 라이브러리가 설치되어 있지 않습니다. 'pip install reportlab'로 설치하세요.")
            return None
        except Exception as e:
            print(f"PDF 보고서 생성 오류: {e}")
            return None