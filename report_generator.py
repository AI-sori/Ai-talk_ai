# report_generator.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import matplotlib.pyplot as plt
import io
import datetime
import numpy as np

# 한글 폰트 등록 (맑은 고딕 폰트 사용)
try:
    pdfmetrics.registerFont(TTFont('Malgun', 'c:/windows/fonts/malgun.ttf'))
except:
    try:
        pdfmetrics.registerFont(TTFont('Malgun', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
    except:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

class PDFReportGenerator:
    def __init__(self, output_path="reports"):
        # 보고서 저장 디렉토리 생성
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 스타일 설정
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Title'],
            fontName='Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
            fontSize=24,
            alignment=1,  # 중앙 정렬
            spaceAfter=20
        )
        
        self.heading_style = ParagraphStyle(
            'Heading',
            parent=self.styles['Heading2'],
            fontName='Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10
        )
        
        self.normal_style = ParagraphStyle(
            'Normal',
            parent=self.styles['Normal'],
            fontName='Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
            fontSize=12,
            spaceBefore=5,
            spaceAfter=5
        )
        
        self.bullet_style = ParagraphStyle(
            'Bullet',
            parent=self.normal_style,
            leftIndent=20,
            spaceBefore=2,
            spaceAfter=2
        )
    
    def generate_pdf(self, diagnostic_data, fixations, child_name="홍길동"):
        """진단 데이터를 기반으로 PDF 보고서 생성"""
        # 현재 날짜로 파일명 생성
        today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_path}/{child_name}_{today}.pdf"
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # 문서에 들어갈 요소 목록
        elements = []
        
        # 제목 추가
        elements.append(Paragraph("읽기 능력 진단 리포트", self.title_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # 기본 정보 테이블
        today_str = datetime.datetime.now().strftime("%Y.%m.%d")
        data = [
            ["아이 이름:", child_name],
            ["진단 날짜:", today_str],
            ["총 읽기 시간:", self._format_time(diagnostic_data.get('session_duration', 0))]
        ]
        
        info_table = Table(data, colWidths=[4*cm, 10*cm])
        info_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12)
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 1*cm))
        
        # 진단 결과 섹션
        elements.append(Paragraph("📌 오늘의 분석 결과", self.heading_style))
        
        # 읽기 지표 테이블
        reading_speed = diagnostic_data.get('reading_speed', '적당함')
        focus_level = diagnostic_data.get('focus_level', '보통')
        comprehension = diagnostic_data.get('comprehension', '보통')
        
        # 이모지 추가
        speed_emoji = "🚀" if reading_speed == "빠름" else "🏃" if reading_speed == "적당함" else "🐢"
        focus_emoji = "🔵" if focus_level == "높음" else "🟠" if focus_level == "보통" else "🔴"
        comp_emoji = "👍" if comprehension == "매우 좋음" else "🙂" if comprehension == "보통" else "🤔"
        
        metrics_data = [
            ["읽기 속도:", f"{speed_emoji} {reading_speed}"],
            ["집중력:", f"{focus_emoji} {focus_level}"],
            ["이해력:", f"{comp_emoji} {comprehension}"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[4*cm, 10*cm])
        metrics_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12)
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.5*cm))
        
        # 시선 추적 결과
        elements.append(Paragraph("📊 시선 추적 결과", self.heading_style))
        
        main_issue = diagnostic_data.get('main_issue', '특별한 문제점이 발견되지 않음')
        elements.append(Paragraph(f"아이가 글을 읽을 때, <b>{main_issue}</b> 현상이 보였어요.", self.normal_style))
        
        # 집중 시간
        session_duration = diagnostic_data.get('session_duration', 0)
        focus_time = diagnostic_data.get('focus_time', session_duration * 0.7)  # 기본값 70%
        
        elements.append(Paragraph(
            f"집중 시간: 총 {self._format_time(session_duration)} 중 {self._format_time(focus_time)} 집중",
            self.normal_style
        ))
        
        # 시선 경로 시각화 (있는 경우)
        if fixations and len(fixations) > 5:
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph("시선 경로 분석", self.normal_style))
            
            # 시선 경로 이미지 생성
            img_data = self._create_scanpath_image(fixations)
            if img_data:
                img = Image(img_data, width=15*cm, height=10*cm)
                elements.append(img)
        
        elements.append(Spacer(1, 0.5*cm))
        
        # 추천 학습 방법
        elements.append(Paragraph("💡 추천 학습 방법", self.heading_style))
        
        # 한 줄 피드백
        feedback = diagnostic_data.get('oneliner_feedback', '꾸준한 독서로 좋은 습관을 유지해보세요')
        elements.append(Paragraph(f"오늘의 한 줄 피드백:", self.normal_style))
        elements.append(Paragraph(f"<i>\"{feedback}\"</i>", self.normal_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # 추천 활동
        elements.append(Paragraph("추천 활동:", self.normal_style))
        
        recommendations = diagnostic_data.get('recommendations', [])
        if not recommendations:
            recommendations = ["규칙적인 독서 시간 – 매일 15분씩 책 읽는 습관 기르기"]
        
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.bullet_style))
        
        # 다음 진단 날짜
        elements.append(Spacer(1, 1*cm))
        next_date = diagnostic_data.get('next_diagnosis_date', 
                                       (datetime.datetime.now() + datetime.timedelta(days=14)).strftime('%Y.%m.%d'))
        elements.append(Paragraph(f"📌 다음 진단 추천일: {next_date}", self.normal_style))

        # PDF 생성
        doc.build(elements)
        print(f"PDF 보고서가 생성되었습니다: {filename}")
        
        # PDF 자동으로 열기
        try:
            import os
            import platform
            import subprocess
            
            if platform.system() == 'Windows':
                os.startfile(filename)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', filename))
            else:  # Linux
                subprocess.call(('xdg-open', filename))
            
            print("PDF 파일이 자동으로 열렸습니다.")
        except Exception as e:
            print(f"PDF 파일을 자동으로 열지 못했습니다: {e}")
        
        return filename
        
    
    def _format_time(self, seconds):
        """초를 분:초 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초"
    
    def _create_scanpath_image(self, fixations):
        """시선 경로 시각화 이미지 생성"""
        try:
            # 시선 좌표 추출
            x_coords = [fx for _, _, fx, _ in fixations]
            y_coords = [fy for _, _, _, fy in fixations]
            
            # 플롯 생성
            plt.figure(figsize=(10, 6))
            plt.scatter(x_coords, y_coords, s=30, c=range(len(x_coords)), cmap='viridis', alpha=0.7)
            
            # 시선 이동 경로 표시
            for i in range(1, len(x_coords)):
                plt.plot([x_coords[i-1], x_coords[i]], [y_coords[i-1], y_coords[i]], 'r-', alpha=0.5)
            
            # 축 설정
            plt.xlim(min(x_coords) - 50, max(x_coords) + 50)
            plt.ylim(max(y_coords) + 50, min(y_coords) - 50)  # y축 반전
            plt.title('시선 경로 분석')
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            
            # 이미지로 변환
            img_data = io.BytesIO()
            plt.savefig(img_data, format='png')
            img_data.seek(0)
            plt.close()
            
            return img_data
        except Exception as e:
            print(f"시선 경로 이미지 생성 오류: {e}")
            return None