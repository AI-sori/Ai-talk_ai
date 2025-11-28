import urllib.request 
import os

def download_model(model_url: str, local_filename: str) -> str:
    """
    URL에서 모델 파일을 다운로드합니다.
    
    :param model_url: 모델 다운로드 URL (직접 전달)
    :param local_filename: 로컬 저장 파일명
    :return: 다운로드된 모델의 로컬 경로
    """
    local_path = os.path.join(os.getcwd(), 'models', local_filename)
    
    # 1. URL 확인
    if not model_url or not model_url.startswith("http"):
        print(f"[ERROR] 유효하지 않은 URL: {model_url}")
        return None
    
    # 2. 이미 다운로드됐으면 재사용
    if os.path.exists(local_path):
        print(f"[INFO] 모델 파일 존재: {local_path}")
        return local_path
    
    # 3. models 폴더 생성
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 4. 다운로드
    print(f"[INFO] 모델 다운로드 중: {model_url}")
    try:
        urllib.request.urlretrieve(model_url, local_path)
        print(f"[SUCCESS] 다운로드 완료: {local_path}")
        return local_path
    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        return None
