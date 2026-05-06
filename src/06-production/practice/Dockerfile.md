# Docker 이미지 빌드 파일 (Dockerfile)

- 원본 파일: `06-프로덕션으로-가는-길/실습/Dockerfile`
- 동기화 방식: 강의 원본에서 자동 생성

아래는 생략 없이 전체 코드입니다.

```dockerfile
# 6장 4절 실습: ML API용 Dockerfile
#
# 사용법:
#   1. 같은 폴더에 다음 파일들이 있어야 함:
#      - 02_FastAPI_서빙.py (이름을 app.py 로 바꾸기)
#      - models/boston_v1.0.0/ 폴더
#      - requirements.txt
#
#   2. 빌드:
#      docker build -t boston-api:1.0 .
#
#   3. 실행:
#      docker run -d -p 8000:8000 --name boston-api boston-api:1.0
#
#   4. 테스트:
#      curl http://localhost:8000/health
#
#   5. 멈춤:
#      docker stop boston-api
#      docker rm boston-api

# 1. 기본 이미지 (slim으로 작게)
FROM python:3.11-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 시스템 의존성 (필요한 만큼만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 의존성 설치 (캐시 최적화: requirements.txt만 먼저)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 앱 코드와 모델 복사
COPY app.py .
COPY models/ ./models/

# 6. 포트 노출
EXPOSE 8000

# 7. 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# 8. 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
