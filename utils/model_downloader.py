import requests
import os

# 이 함수는 환경 변수에서 모델의 다운로드 URL을 읽어옵니다.
def download_model(env_variable_name: str, local_filename: str) -> str:
    """
    환경 변수에 지정된 URL에서 모델 파일을 다운로드합니다.
    
    :param env_variable_name: 모델 URL을 담고 있는 환경 변수 이름 (예: 'GAZE_MODEL_URL')
    :param local_filename: 다운로드 후 저장할 로컬 파일 이름 (예: 'gaze_model.pth')
    :return: 다운로드된 모델의 로컬 경로
    """
    model_url = os.environ.get(env_variable_name)
    local_path = os.path.join(os.getcwd(), 'models', local_filename)

    # 1. 모델 URL 확인
    if not model_url:
        print(f"[ERROR] 환경 변수 {env_variable_name}이 설정되지 않았습니다. 더미 모드 활성화.")
        # 실제 배포 환경에서는 여기서 오류를 발생시키거나, 사용자님의 더미 로직을 따라야 합니다.
        return None 
    
    # 2. 로컬에 모델 폴더 생성
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 3. 파일이 이미 존재하는지 확인 (재배포 시 불필요한 다운로드 방지)
    if os.path.exists(local_path):
        print(f"[INFO] 모델 파일 {local_filename}이 이미 로컬에 존재합니다. 다운로드를 건너뜁니다.")
        return local_path

    # 4. 파일 다운로드 실행
    print(f"[INFO] 외부 URL에서 모델 다운로드 시작: {model_url}")
    try:
        # stream=True를 사용하여 대용량 파일을 메모리 과부하 없이 청크별로 다운로드
        with requests.get(model_url, stream=True, timeout=300) as r:
            r.raise_for_status() # HTTP 오류 발생 시 예외 처리
            total_size = r.headers.get('content-length')
            
            with open(local_path, 'wb') as f:
                if total_size is None: # Content-Length 헤더가 없는 경우
                    f.write(r.content)
                else:
                    # 다운로드 진행 상황 표시 (선택 사항)
                    downloaded = 0
                    total_size = int(total_size)
                    for chunk in r.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
        
        print(f"[SUCCESS] 모델 파일 {local_filename} 다운로드 완료! 경로: {local_path}")
        return local_path

    except requests.exceptions.RequestException as e:
        print(f"[FATAL ERROR] 모델 다운로드 실패: {e}")
        # 다운로드 실패 시 로컬 파일을 삭제하여 다음 시도 때 재다운로드 유도
        if os.path.exists(local_path):
            os.remove(local_path)
        return None
