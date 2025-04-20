document.addEventListener('DOMContentLoaded', function() {
    // 소켓 연결
    const socket = io();
    
    // HTML 요소 참조
    const videoStream = document.getElementById('video-stream');
    const currentState = document.getElementById('current-state');
    const textContainer = document.getElementById('text-container');
    const textTitle = document.getElementById('text-title');
    const textContent = document.getElementById('text-content');
    const readingInfo = document.getElementById('reading-info');
    const readingTime = document.getElementById('reading-time');
    const diagnosisInfo = document.getElementById('diagnosis-info');
    const remainingTime = document.getElementById('remaining-time');
    const reportContainer = document.getElementById('report-container');
    const generatePdfBtn = document.getElementById('generate-pdf');
    const downloadPdfLink = document.getElementById('download-pdf');
    
    // 버튼 요소 참조
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const startSessionBtn = document.getElementById('start-session');
    const readingStartBtn = document.getElementById('reading-start');
    const diagnosisStartBtn = document.getElementById('diagnosis-start');
    const resetSessionBtn = document.getElementById('reset-session');
    const prevTextBtn = document.getElementById('prev-text');
    const nextTextBtn = document.getElementById('next-text');
    
    // 현재 상태와 타이머
    let currentStateValue = 'IDLE';
    let readingTimerId = null;
    let readingStartTime = 0;
    let diagnosisTimerId = null;
    
    // 상태 변경 함수
    function changeState(newState) {
        socket.emit('change_state', { state: newState });
    }
    
    // 읽기 시간 업데이트 함수
    function updateReadingTime() {
        const elapsed = Math.floor((Date.now() - readingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        readingTime.textContent = `${minutes}분 ${seconds}초`;
    }
    
    // 진단 카운트다운 함수
    function startDiagnosisCountdown() {
        let count = 5;
        remainingTime.textContent = count;
        
        diagnosisTimerId = setInterval(() => {
            count--;
            remainingTime.textContent = count;
            
            if (count <= 0) {
                clearInterval(diagnosisTimerId);
                // 서버에서 진단 완료 알림을 받을 것이므로 여기서는 추가 작업 필요 없음
            }
        }, 1000);
    }
    
    // UI 업데이트 함수
    function updateUI() {
        // 모든 상태별 컨테이너 숨기기
        textContainer.classList.add('hidden');
        readingInfo.classList.add('hidden');
        diagnosisInfo.classList.add('hidden');
        reportContainer.classList.add('hidden');
        
        // 모든 상태별 버튼 숨기기
        readingStartBtn.classList.add('hidden');
        diagnosisStartBtn.classList.add('hidden');
        resetSessionBtn.classList.add('hidden');
        
        // 현재 상태에 따라 요소 표시
        currentState.textContent = currentStateValue;
        
        switch (currentStateValue) {
            case 'IDLE':
                currentState.textContent = '준비';
                break;
                
            case 'TEXT_VIEW':
                currentState.textContent = '텍스트 선택';
                textContainer.classList.remove('hidden');
                readingStartBtn.classList.remove('hidden');
                break;
                
            case 'READING':
                currentState.textContent = '읽기 중';
                textContainer.classList.remove('hidden');
                readingInfo.classList.remove('hidden');
                diagnosisStartBtn.classList.remove('hidden');
                
                // 읽기 시간 타이머 시작
                readingStartTime = Date.now();
                if (readingTimerId) clearInterval(readingTimerId);
                readingTimerId = setInterval(updateReadingTime, 1000);
                break;
                
            case 'DIAGNOSIS':
                currentState.textContent = '진단 중';
                textContainer.classList.remove('hidden');
                diagnosisInfo.classList.remove('hidden');
                
                // 읽기 시간 타이머 중지
                if (readingTimerId) {
                    clearInterval(readingTimerId);
                    readingTimerId = null;
                }
                
                // 진단 카운트다운 시작
                startDiagnosisCountdown();
                break;
                
            case 'REPORT':
                currentState.textContent = '진단 완료';
                reportContainer.classList.remove('hidden');
                resetSessionBtn.classList.remove('hidden');
                
                // 모든 타이머 중지
                if (readingTimerId) {
                    clearInterval(readingTimerId);
                    readingTimerId = null;
                }
                if (diagnosisTimerId) {
                    clearInterval(diagnosisTimerId);
                    diagnosisTimerId = null;
                }
                break;
        }
    }
    
    // 소켓 이벤트 핸들러
    socket.on('connect', function() {
        console.log('서버에 연결됨');
    });
    
    socket.on('video_frame', function(data) {
        videoStream.src = 'data:image/jpeg;base64,' + data.frame;
    });
    
    socket.on('state_changed', function(data) {
        currentStateValue = data.state;
        updateUI();
    });
    
    socket.on('text_data', function(data) {
        textTitle.textContent = data.text.title;
        textContent.textContent = data.text.content;
    });
    
    socket.on('diagnosis_complete', function(data) {
        console.log('진단 완료:', data);
        // 진단 결과를 표시하거나 처리할 수 있음
    });
    
    socket.on('pdf_generated', function(data) {
        // PDF 다운로드 링크 업데이트
        downloadPdfLink.href = '/download_pdf/' + data.path.split('/').pop();
        downloadPdfLink.classList.remove('hidden');
    });
    
    // 버튼 이벤트 핸들러
    startCameraBtn.addEventListener('click', function() {
        socket.emit('start_camera');
    });
    
    stopCameraBtn.addEventListener('click', function() {
        socket.emit('stop_camera');
    });
    
    startSessionBtn.addEventListener('click', function() {
        changeState('TEXT_VIEW');
    });
    
    readingStartBtn.addEventListener('click', function() {
        changeState('READING');
    });
    
    diagnosisStartBtn.addEventListener('click', function() {
        changeState('DIAGNOSIS');
    });
    
    resetSessionBtn.addEventListener('click', function() {
        changeState('IDLE');
    });
    
    prevTextBtn.addEventListener('click', function() {
        socket.emit('prev_text');
    });
    
    nextTextBtn.addEventListener('click', function() {
        socket.emit('next_text');
    });
    
    generatePdfBtn.addEventListener('click', function() {
        socket.emit('generate_pdf');
    });
});