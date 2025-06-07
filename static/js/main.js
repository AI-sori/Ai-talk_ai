// 전역 변수
let video, canvas, ctx;
let isTracking = false;
let calibrationStep = 0;
let trackingInterval;
let startTime;
let mediaRecorder;
let audioChunks = [];
let audioResult = null;

// 다중 이야기 시스템
let currentStory = 0;
let allTrackingData = [];
let storyStartTime = 0;

const stories = [
    {
        title: "토끼와 거북이",
        text: `🐰 토끼와 거북이가 달리기를 했어요. 
토끼는 빨리 뛰어갔지만 중간에 잠을 잤어요. 💤
거북이는 천천히 걸어갔어요. 🐢
결국 거북이가 먼저 도착했어요! 🏆
"천천히 해도 끝까지 하면 이길 수 있어요!"`
    },
    {
        title: "개미와 베짱이",
        text: `🐜 개미는 여름에 열심히 일했어요.
🦗 베짱이는 노래만 불렀어요. 🎵
겨울이 되자 개미는 따뜻한 집에서 지냈어요. 🏠
베짱이는 춥고 배가 고팠어요. ❄️
"미리미리 준비하는 것이 중요해요!"`
    },
    {
        title: "아기돼지 삼형제",
        text: `🐷 아기돼지 삼형제가 집을 지었어요.
첫째는 짚으로, 둘째는 나무로 지었어요. 🏘️
셋째는 벽돌로 튼튼하게 지었어요. 🧱
늑대가 와서 후~ 불었지만 벽돌집만 안전했어요! 🐺
"튼튼하게 만드는 것이 최고예요!"`
    }
];

// DOM 요소들
const el = {
    status: document.getElementById('status'),
    initBtn: document.getElementById('initBtn'),
    calibrateBtn: document.getElementById('calibrateBtn'),
    startTrackingBtn: document.getElementById('startTrackingBtn'),
    stopTrackingBtn: document.getElementById('stopTrackingBtn'),
    readingArea: document.getElementById('readingArea'),
    trackingInfo: document.getElementById('trackingInfo'),
    audioSection: document.getElementById('audioSection'),
    childInfoSection: document.getElementById('childInfoSection'),
    reportSection: document.getElementById('reportSection'),
    currentDirection: document.getElementById('currentDirection'),
    confidence: document.getElementById('confidence'),
    errorOffset: document.getElementById('errorOffset'),
    trackingTime: document.getElementById('trackingTime'),
    startRecordBtn: document.getElementById('startRecordBtn'),
    stopRecordBtn: document.getElementById('stopRecordBtn'),
    recordingStatus: document.getElementById('recordingStatus'),
    childName: document.getElementById('childName'),
    userId: document.getElementById('userId'),
    generateReportBtn: document.getElementById('generateReportBtn'),
    reportContent: document.getElementById('reportContent'),
    downloadReportBtn: document.getElementById('downloadReportBtn'),
    readingText: document.getElementById('readingText'),
    storyCounter: document.getElementById('storyCounter'),
    nextStoryBtn: document.getElementById('nextStoryBtn'),
    prevStoryBtn: document.getElementById('prevStoryBtn')
};

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeCamera();
    setupEventListeners();
});

// 이벤트 리스너 설정 (안전하게)
function setupEventListeners() {
    // 필수 요소들만 체크
    if (el.initBtn) el.initBtn.addEventListener('click', initializeSystem);
    if (el.calibrateBtn) el.calibrateBtn.addEventListener('click', startCalibration);
    if (el.startTrackingBtn) el.startTrackingBtn.addEventListener('click', startTracking);
    if (el.stopTrackingBtn) el.stopTrackingBtn.addEventListener('click', stopTracking);
    if (el.startRecordBtn) el.startRecordBtn.addEventListener('click', startRecording);
    if (el.stopRecordBtn) el.stopRecordBtn.addEventListener('click', stopRecording);
    if (el.generateReportBtn) el.generateReportBtn.addEventListener('click', generateReport);
    if (el.downloadReportBtn) el.downloadReportBtn.addEventListener('click', downloadPDFReport);
    if (el.nextStoryBtn) el.nextStoryBtn.addEventListener('click', nextStory);
    
    console.log('[INFO] 이벤트 리스너 설정 완료');
}

// 상태 업데이트
function updateStatus(message, type = 'info') {
    el.status.textContent = message;
    el.status.className = `status ${type}`;
}

// 카메라 초기화
async function initializeCamera() {
    try {
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 },
            audio: true 
        });
        
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            updateStatus('카메라 준비 완료. 시스템을 초기화하세요.', 'info');
        };
    } catch (error) {
        updateStatus('카메라 접근 실패', 'error');
    }
}

// 시스템 초기화
async function initializeSystem() {
    try {
        updateStatus('시스템 초기화 중...', 'info');
        el.initBtn.disabled = true;
        
        const response = await fetch('/init_tracker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            updateStatus('초기화 완료! 보정을 시작하세요.', 'success');
            el.calibrateBtn.disabled = false;
        } else {
            updateStatus('초기화 실패', 'error');
            el.initBtn.disabled = false;
        }
    } catch (error) {
        updateStatus('초기화 오류', 'error');
        el.initBtn.disabled = false;
    }
}

// 보정 시작
function startCalibration() {
    calibrationStep = 0;
    updateStatus('보정 시작. 빨간 점을 응시하세요.', 'info');
    el.calibrateBtn.disabled = true;
    showCalibrationPoint();
}

// 보정 포인트 표시
function showCalibrationPoint() {
    const points = [
        { x: 200, y: 200 },
        { x: window.innerWidth - 200, y: 200 },
        { x: 200, y: window.innerHeight - 200 },
        { x: window.innerWidth - 200, y: window.innerHeight - 200 },
        { x: window.innerWidth / 2, y: window.innerHeight / 2 }
    ];
    
    if (calibrationStep >= points.length) {
        updateStatus('보정 완료! 추적을 시작할 수 있습니다.', 'success');
        el.startTrackingBtn.disabled = false;
        return;
    }
    
    const point = points[calibrationStep];
    
    // 보정 점 생성
    const dot = document.createElement('div');
    dot.style.cssText = `
        position: fixed;
        left: ${point.x - 15}px;
        top: ${point.y - 15}px;
        width: 30px;
        height: 30px;
        background: red;
        border-radius: 50%;
        z-index: 9999;
        animation: pulse 1s infinite;
    `;
    
    document.body.appendChild(dot);
    
    updateStatus(`보정 ${calibrationStep + 1}/5`, 'info');
    
    // 3초 후 보정 실행
    setTimeout(async () => {
        await performCalibration(point.x, point.y);
        document.body.removeChild(dot);
        calibrationStep++;
        setTimeout(showCalibrationPoint, 500);
    }, 3000);
}

// 보정 실행
async function performCalibration(targetX, targetY) {
    try {
        const frameData = captureFrame();
        const response = await fetch('/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame: frameData,
                target_x: targetX,
                target_y: targetY
            })
        });
        
        const result = await response.json();
        console.log('보정 결과:', result);
    } catch (error) {
        console.error('보정 오류:', error);
    }
}

// 프레임 캡처
function captureFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// 추적 시작
async function startTracking() {
    try {
        const response = await fetch('/start_tracking', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            isTracking = true;
            startTime = Date.now();
            storyStartTime = Date.now();
            currentStory = 0;
            allTrackingData = [];
            
            updateStatus('시선 추적 시작!', 'success');
            
            el.startTrackingBtn.disabled = true;
            el.stopTrackingBtn.disabled = false;
            el.readingArea.style.display = 'block';
            el.trackingInfo.style.display = 'block';
            el.audioSection.style.display = 'block';  // 음성 섹션 바로 표시
            
            showCurrentStory();
            startGazeTracking();
            
            // 20초 후 다음 버튼 표시
            setTimeout(() => {
                if (currentStory < 2) {
                    el.nextStoryBtn.style.display = 'inline-block';
                }
            }, 20000);
            
        } else {
            updateStatus('추적 시작 실패', 'error');
        }
    } catch (error) {
        updateStatus('추적 오류', 'error');
    }
}

// 현재 이야기 표시
function showCurrentStory() {
    const story = stories[currentStory];
    el.readingText.innerHTML = story.text;
    el.storyCounter.textContent = `(${currentStory + 1}/3)`;
    
    // 다음 버튼 숨기기
    el.nextStoryBtn.style.display = 'none';
    
    if (currentStory < 2) {
        // 20초 후 다음 버튼 표시
        setTimeout(() => {
            el.nextStoryBtn.style.display = 'inline-block';
            updateStatus(`다음 이야기로 넘어가세요!`, 'info');
        }, 20000);
    }
}

// 다음 이야기
function nextStory() {
    if (currentStory < 2) {
        currentStory++;
        storyStartTime = Date.now();
        showCurrentStory();
        updateStatus(`${currentStory + 1}번째 이야기를 읽어보세요!`, 'info');
    } else {
        updateStatus('모든 이야기 완료!', 'success');
        el.nextStoryBtn.style.display = 'none';
        el.childInfoSection.style.display = 'block';
    }
}

// 시선 추적 루프
function startGazeTracking() {
    trackingInterval = setInterval(async () => {
        if (!isTracking) return;
        
        try {
            const frameData = captureFrame();
            const response = await fetch('/track_gaze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: frameData })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // 현재 이야기 정보 추가
                result.story = currentStory + 1;
                result.timestamp = Date.now();
                
                // 전체 데이터에 추가
                allTrackingData.push(result);
                
                updateTrackingInfo(result);
            }
        } catch (error) {
            console.error('추적 오류:', error);
        }
    }, 500);
}

// 추적 정보 업데이트
function updateTrackingInfo(result) {
    const colors = { 'left': '#ff6b6b', 'center': '#4ecdc4', 'right': '#45b7d1' };
    
    el.currentDirection.textContent = result.direction || '-';
    el.currentDirection.style.color = colors[result.direction] || '#333';
    
    el.confidence.textContent = result.confidence ? 
        (result.confidence * 100).toFixed(1) + '%' : '-';
    
    el.errorOffset.textContent = result.error_offset ? 
        result.error_offset.toFixed(1) + 'px' : '-';
    
    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        el.trackingTime.textContent = elapsed + '초';
    }
}

// 추적 중지
async function stopTracking() {
    isTracking = false;
    if (trackingInterval) {
        clearInterval(trackingInterval);
    }
    
    await fetch('/stop_tracking', { method: 'POST' });
    
    updateStatus('추적 중지. 음성 녹음을 진행하세요.', 'info');
    el.startTrackingBtn.disabled = false;
    el.stopTrackingBtn.disabled = true;
    el.childInfoSection.style.display = 'block';
}

// 녹음 시작
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await analyzeAudio(audioBlob);
        };
        
        mediaRecorder.start();
        
        el.startRecordBtn.disabled = true;
        el.stopRecordBtn.disabled = false;
        el.recordingStatus.textContent = '🔴 녹음 중...';
        updateStatus('음성 녹음 시작', 'info');
        
    } catch (error) {
        updateStatus('마이크 접근 실패', 'error');
    }
}

// 녹음 중지
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        
        el.startRecordBtn.disabled = false;
        el.stopRecordBtn.disabled = true;
        el.recordingStatus.textContent = '🔄 분석 중...';
        updateStatus('음성 분석 중...', 'info');
    }
}

// 음성 분석
async function analyzeAudio(audioBlob) {
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        const response = await fetch('/analyze_audio', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            audioResult = result.result;
            el.recordingStatus.textContent = '✅ 분석 완료';
            updateStatus('음성 분석 완료!', 'success');
        } else {
            el.recordingStatus.textContent = '❌ 분석 실패';
            updateStatus('음성 분석 실패', 'error');
        }
    } catch (error) {
        el.recordingStatus.textContent = '❌ 오류';
        updateStatus('음성 분석 오류', 'error');
    }
}

// 리포트 생성
async function generateReport() {
    try {
        const childName = el.childName.value.trim();
        const userId = parseInt(el.userId.value) || 1;
        
        if (!childName) {
            updateStatus('아동 이름을 입력하세요', 'error');
            return;
        }
        
        updateStatus('리포트 생성 중...', 'info');
        el.generateReportBtn.disabled = true;
        
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                child_name: childName,
                user_id: userId,
                audio_result: audioResult || {}
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayReport(result.report);
            updateStatus('리포트 생성 완료!', 'success');
        } else {
            updateStatus('리포트 생성 실패', 'error');
            el.generateReportBtn.disabled = false;
        }
    } catch (error) {
        updateStatus('리포트 생성 오류', 'error');
        el.generateReportBtn.disabled = false;
    }
}

// 통합 데이터 분석
function analyzeCombinedData() {
    if (allTrackingData.length === 0) return null;
    
    const directions = { left: 0, center: 0, right: 0 };
    let totalConfidence = 0;
    
    allTrackingData.forEach(data => {
        directions[data.direction]++;
        totalConfidence += data.confidence || 0.5;
    });
    
    const avgConfidence = totalConfidence / allTrackingData.length;
    const totalTime = allTrackingData.length * 0.5; // 0.5초 간격
    
    return {
        directions,
        avgConfidence,
        totalTime,
        totalMeasurements: allTrackingData.length
    };
}

// 리포트 표시 (적당한 디테일)
function displayReport(report) {
    const r = report.report;
    
    // 데이터 파싱
    const concentration = parseFloat(r.results.concentration.replace('%', ''));
    const comprehension = parseFloat(r.results.comprehension.replace('%', ''));
    const readingSpeed = parseFloat(r.results.reading_speed.split(' ')[0]);
    const fluency = parseFloat(r.speech_analysis.fluency.replace('%', ''));
    const clarity = parseFloat(r.speech_analysis.pronunciation_clarity.replace('%', ''));
    
    // 이모지와 레벨 결정
    const concentrationLevel = concentration >= 70 ? '🔵 높음' : concentration >= 40 ? '🟠 보통' : '🔴 주의필요';
    const comprehensionLevel = comprehension >= 80 ? '👍 매우좋음' : comprehension >= 60 ? '🙂 보통' : '🤔 연습필요';
    const speedLevel = readingSpeed >= 50 ? '🚀 빠름' : readingSpeed >= 20 ? '🏃 적당' : '🐢 느림';
    const fluencyLevel = fluency >= 80 ? '✨ 매우좋음' : fluency >= 60 ? '👌 좋음' : '📝 연습필요';
    const clarityLevel = clarity >= 80 ? '🎯 명확함' : clarity >= 60 ? '👂 괜찮음' : '🗣️ 연습필요';
    
    // 읽기 시간 변환
    const totalSeconds = parseFloat(r.reading_time.replace('초', ''));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const timeText = minutes > 0 ? `${minutes}분${seconds}초` : `${seconds}초`;
    
    // 집중 시간
    const focusSeconds = parseFloat(r.eye_tracking.focus_time.replace('초', ''));
    const focusText = focusSeconds > 0 ? `${focusSeconds.toFixed(0)}초` : '0초';
    
    // 종합 피드백
    let feedback = "";
    if (concentration < 40) {
        feedback = "집중력 연습이 필요해요! 짧은 시간부터 시작해보세요 📚";
    } else if (comprehension < 60) {
        feedback = "이해력을 키우는 연습을 해보세요! 질문하며 읽어보세요 💭";
    } else if (fluency < 60) {
        feedback = "발음과 말하기 연습을 더 해보세요! 천천히 또박또박 📢";
    } else {
        feedback = "정말 잘하고 있어요! 계속 꾸준히 읽어보세요 ⭐";
    }
    
    const balancedReport = `
📚 ${r.child_name}의 읽기 분석 리포트

📅 진단날짜: ${r.diagnosis_date.replace(/-/g, '.')}
⏰ 총 읽기시간: ${timeText} (집중시간: ${focusText})

📊읽기 능력 분석
• 읽기속도: ${speedLevel}
• 집중력: ${concentrationLevel}
• 이해력: ${comprehensionLevel}

🎤 음성 분석 결과
• 발음 명확도: ${clarityLevel} (${r.speech_analysis.pronunciation_clarity})
• 말하기 유창성: ${fluencyLevel} (${r.speech_analysis.fluency})
• 말하기 속도: ${r.speech_analysis.speaking_rate}

💬 아이가 말한 내용:
"${r.speech_analysis.transcription}"

👀 시선 패턴:
${r.eye_tracking.issues === '정상' ? '✅ 자연스러운 시선 움직임을 보였어요!' : `⚠️ ${r.eye_tracking.issues} 현상이 관찰되었어요.`}

💡 종합 피드백
${feedback}

📝 맞춤 추천 활동
${r.feedback.recommended_activities.slice(0, 4).map(activity => `• ${activity}`).join('\n')}

📌 다음 검사 추천일: ${r.feedback.next_diagnosis_date.replace(/-/g, '.')}
    `.trim();
    
    el.reportContent.textContent = balancedReport;
    el.reportContent.style.fontFamily = 'Arial, sans-serif';
    el.reportContent.style.fontSize = '14px';
    el.reportContent.style.lineHeight = '1.6';
    el.reportContent.style.background = '#f8f9fa';
    el.reportContent.style.border = '2px solid #e9ecef';
    el.reportContent.style.borderRadius = '8px';
    el.reportContent.style.padding = '20px';
    
    el.reportSection.style.display = 'block';
    el.downloadReportBtn.disabled = false;
}

// 리포트 다운로드
// main.js에서 downloadPDFReport 함수 수정
async function downloadPDFReport() {
    const childName = el.childName.value.trim();
    const userId = parseInt(el.userId.value) || 1;
    
    if (!childName) {
        updateStatus('아동 이름을 입력해주세요.', 'error');
        return;
    }
    
    const audioData = audioResult || {
        transcription: '음성 분석 결과 없음',
        fluency: '0.0%',
        pronunciation_clarity: '0.0%',
        comprehension: '0.0%',
        speaking_rate: '0.0 단어/분',
        duration: '0.0초',
        word_count: 0
    };
    
    try {
        updateStatus('📄 PDF 생성 중...', 'info');
        el.downloadReportBtn.disabled = true;
        
        console.log('[DEBUG] 요청 데이터:', {
            child_name: childName,
            user_id: userId,
            audio_result: audioData
        });
        
        const response = await fetch('/download_pdf_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                child_name: childName,
                user_id: userId,
                audio_result: audioData
            })
        });
        
        console.log('[DEBUG] 응답 상태:', response.status);
        console.log('[DEBUG] 응답 헤더:', response.headers);
        
        // 🔥 응답을 텍스트로 먼저 받아서 확인
        const responseText = await response.text();
        console.log('[DEBUG] 응답 내용:', responseText);
        
        // JSON 파싱 시도
        let result;
        try {
            result = JSON.parse(responseText);
            console.log('[DEBUG] 파싱된 JSON:', result);
        } catch (parseError) {
            console.error('[ERROR] JSON 파싱 실패:', parseError);
            console.error('[ERROR] 응답 텍스트:', responseText);
            throw new Error('서버 응답이 JSON 형식이 아닙니다');
        }
        
        if (result.status === 'success') {
            console.log('[DEBUG] PDF 데이터 길이:', result.pdf_data ? result.pdf_data.length : 'undefined');
            
            // Base64를 Blob으로 변환
            const pdfData = atob(result.pdf_data);
            const pdfArray = new Uint8Array(pdfData.length);
            for (let i = 0; i < pdfData.length; i++) {
                pdfArray[i] = pdfData.charCodeAt(i);
            }
            const pdfBlob = new Blob([pdfArray], { type: 'application/pdf' });
            
            // 다운로드
            const downloadUrl = URL.createObjectURL(pdfBlob);
            const downloadLink = document.createElement('a');
            downloadLink.href = downloadUrl;
            downloadLink.download = result.filename || 'report.pdf';
            downloadLink.click();
            
            URL.revokeObjectURL(downloadUrl);
            updateStatus('✅ PDF 다운로드 완료!', 'success');
            
        } else {
            console.error('[ERROR] 서버 오류:', result.message);
            throw new Error(result.message || 'PDF 생성 실패');
        }
        
    } catch (error) {
        console.error('[ERROR] PDF 다운로드 상세 오류:', error);
        updateStatus(`❌ PDF 생성 실패: ${error.message}`, 'error');
    } finally {
        el.downloadReportBtn.disabled = false;
    }
}

// 기존 generateReport 함수에서 PDF 버튼 활성화 부분만 수정
// 리포트 생성 완료 후 이 부분 추가:
const downloadBtn = document.getElementById('downloadReportBtn');
downloadBtn.disabled = false;
downloadBtn.onclick = downloadPDFReport;  // 이 줄만 변경
// CSS 추가
const style = document.createElement('style');
style.textContent = `
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}
.status.info { background: #e3f2fd; color: #1976d2; }
.status.success { background: #e8f5e8; color: #2e7d32; }
.status.error { background: #ffebee; color: #c62828; }
`;
document.head.appendChild(style);