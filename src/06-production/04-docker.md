# 6.4 Docker로 패키징

## "내 컴퓨터에선 되는데..."

API 서버를 만들었어요. 본인 컴퓨터에서 잘 돌아가요. 그런데 회사 서버에 올렸더니 안 돌아가요. 왜?

- Python 버전이 다름
- 라이브러리 버전이 다름
- OS가 다름
- 시스템 라이브러리가 없음

이걸 해결하는 게 **Docker**예요. Docker는 **앱과 그 환경을 통째로 패키징**해서, 어디서든 똑같이 돌아가게 해 줍니다.

---

## Docker가 뭐예요?

비유하자면, 컨테이너 박스예요.

```
일반 배포:
   Python 3.9 + 라이브러리 + 코드 → 회사 서버 (Python 3.7? 라이브러리 없음?)
                                    → 충돌, 에러

Docker 배포:
   Python 3.9 + 라이브러리 + 코드 + 미니 OS  → [컨테이너로 패키징]
                                              → 회사 서버 (어떤 환경이든)
                                              → 정상 동작!
```

핵심 아이디어: **앱이 필요한 모든 것을 한 박스에 넣어서 통째로 배송.**

---

## Docker 설치

### 본인 컴퓨터

- **Windows / Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 패키지 매니저로 (apt, yum 등)

설치 후 확인:

```bash
docker --version
# Docker version 24.0.0
```

---

## Dockerfile: 박스의 설계도

Dockerfile은 **박스를 어떻게 만들지** 적은 텍스트 파일이에요.

ML API용 Dockerfile 예시:

```dockerfile
# Dockerfile

# 1. 기본 이미지 (이미 만들어진 박스에서 시작)
FROM python:3.11-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 의존성 먼저 설치 (캐시 효율성)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 코드 복사
COPY app.py .
COPY models/ ./models/

# 5. 포트 열기
EXPOSE 8000

# 6. 실행 명령
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

각 줄이 박스를 만드는 한 단계예요. 위에서 아래로 실행됩니다.

### 한 줄씩 설명

#### `FROM python:3.11-slim`
**기본 박스를 가져와요.** `python:3.11-slim`은 가벼운 Python 3.11이 설치된 박스. Docker Hub에서 자동 다운로드.

#### `WORKDIR /app`
**박스 안의 작업 폴더를 정함.** 이후 명령은 모두 `/app`에서 실행돼요.

#### `COPY requirements.txt .`
**우리 컴퓨터의 파일을 박스로 복사.**

#### `RUN pip install ...`
**박스 안에서 명령 실행.** Python 라이브러리 설치.

#### `COPY app.py .` / `COPY models/ ./models/`
앱 코드와 모델 파일을 박스로 복사.

#### `EXPOSE 8000`
"이 박스는 8000 포트를 쓴다" 라고 알림.

#### `CMD [...]`
**박스가 시작될 때 실행할 명령.** uvicorn으로 우리 API 시작.

---

## requirements.txt

박스에 들어갈 라이브러리 목록.

```
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.0
scikit-learn==1.3.0
joblib==1.3.0
numpy==1.26.0
```

**버전을 정확히 명시**하세요. 안 그러면 6개월 후에 어떤 버전이 설치될지 몰라요.

---

## 박스 만들기 (build)

```bash
docker build -t my-ml-api:1.0 .
```

- `-t my-ml-api:1.0`: 이름과 태그
- `.`: 현재 폴더의 Dockerfile

빌드가 끝나면 박스(이미지)가 생겨요. 확인:

```bash
docker images
# REPOSITORY     TAG    IMAGE ID       CREATED         SIZE
# my-ml-api      1.0    abc123def456   2 minutes ago   500MB
```

---

## 박스 실행 (run)

```bash
docker run -p 8000:8000 my-ml-api:1.0
```

- `-p 8000:8000`: 호스트 포트 : 컨테이너 포트
- `my-ml-api:1.0`: 실행할 이미지

브라우저에서 `http://localhost:8000`으로 가시면 API가 동작해요. **정확히 본인 컴퓨터에서 직접 실행한 거랑 똑같아 보여요.**

차이는: **이 컨테이너를 그대로 다른 컴퓨터/서버로 옮겨도 잘 돌아간다는 것.**

### 백그라운드 실행

```bash
docker run -d -p 8000:8000 --name my-api my-ml-api:1.0
```

- `-d`: 백그라운드 (detached)
- `--name my-api`: 컨테이너 이름

실행 중인 컨테이너 보기:

```bash
docker ps
```

멈추기:

```bash
docker stop my-api
```

---

## 작은 박스가 좋은 박스

기본 Python 이미지(`python:3.11`)는 약 1GB예요. 너무 커요. 다음 옵션들로 줄일 수 있어요.

### 옵션 1: slim 버전

```dockerfile
FROM python:3.11-slim   # 약 150MB
```

`-slim`은 불필요한 시스템 도구를 뺀 버전. **거의 항상 이걸 쓰세요.**

### 옵션 2: alpine

```dockerfile
FROM python:3.11-alpine   # 약 50MB
```

가장 작지만, ML 라이브러리(numpy, scipy 등)가 잘 안 돌아갈 수 있어요. **ML에는 권장 X.**

### 옵션 3: 멀티스테이지 빌드

```dockerfile
# 1단계: 빌드
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 2단계: 실행 (빌드 도구는 빼고)
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app.py .
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

빌드 단계와 실행 단계를 분리해서 최종 이미지를 작게.

---

## .dockerignore

빌드할 때 박스에 안 넣을 파일을 지정.

```
# .dockerignore
__pycache__/
*.pyc
.git/
.env
*.ipynb
data/
notebooks/
.pytest_cache/
```

이걸 안 하면 박스가 매우 커져요. **꼭 만드세요.**

---

## GPU 사용 (PyTorch 모델)

GPU가 필요한 모델이라면 NVIDIA용 베이스 이미지를 쓰세요.

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Python 설치
RUN apt-get update && apt-get install -y python3.11 python3.11-pip

# 이후는 동일
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

실행 시 GPU 옵션:

```bash
docker run --gpus all -p 8000:8000 my-ml-api:1.0
```

---

## docker-compose: 여러 서비스 한꺼번에

API 서버 + 데이터베이스 + Redis 같이 쓰면 `docker-compose.yml`이 편해요.

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

실행:

```bash
docker-compose up -d
```

---

## Docker Hub에 올리기 (배포)

본인이 만든 이미지를 인터넷에 공유.

```bash
# 1. Docker Hub 로그인
docker login

# 2. 이미지 태그
docker tag my-ml-api:1.0 yourusername/my-ml-api:1.0

# 3. 푸시
docker push yourusername/my-ml-api:1.0
```

다른 사람이 사용:

```bash
docker pull yourusername/my-ml-api:1.0
docker run -p 8000:8000 yourusername/my-ml-api:1.0
```

---

## 클라우드에 배포

만들어진 Docker 이미지는 클라우드 서비스에 올려서 자동으로 운영할 수 있어요.

| 클라우드 | 서비스 |
|---------|------|
| AWS | ECS, EKS, App Runner |
| GCP | Cloud Run, GKE |
| Azure | ACI, AKS |
| 기타 | Heroku, Railway, Fly.io, Render |

가장 단순한 길은 **Cloud Run**(GCP) 또는 **App Runner**(AWS)예요. Docker 이미지만 주면 자동 배포 + 자동 스케일링.

---

## 디버깅 팁

### 컨테이너 안에 들어가 보기

```bash
docker exec -it my-api bash
```

박스 안의 셸에 접속. 파일 확인, 로그 확인 등 가능.

### 로그 보기

```bash
docker logs my-api
docker logs -f my-api    # 실시간
```

### 자원 사용량

```bash
docker stats
```

CPU, 메모리 사용률 실시간.

---

## 정리

```bash
# 1. Dockerfile 만들기
# 2. requirements.txt 만들기
# 3. .dockerignore 만들기
# 4. 빌드
docker build -t my-ml-api:1.0 .

# 5. 실행
docker run -d -p 8000:8000 --name my-api my-ml-api:1.0

# 6. 확인
docker logs my-api
curl http://localhost:8000/predict ...
```

**한 줄 요약: "Dockerfile 한 파일로 모델을 어디서든 돌아가게."**

---

## 자주 묻는 질문

> **Q. Docker 꼭 써야 해요?**
>
> 작은 프로젝트면 안 써도 됩니다. 하지만 회사 환경에서는 거의 표준이에요. 한 번 익히시면 평생 갑니다.

> **Q. 박스 크기가 너무 커요. 어떻게 줄여요?**
>
> - `-slim` 또는 `-alpine` 베이스 이미지
> - .dockerignore에 불필요한 파일 추가
> - 멀티스테이지 빌드
> - 모델 파일을 외부 저장소(S3 등)에서 가져오기

> **Q. Docker와 Kubernetes 차이가 뭐예요?**
>
> - Docker: 컨테이너 만드는 도구
> - Kubernetes: 여러 컨테이너 관리 (배포, 스케일링, 복구)
>
> Kubernetes는 큰 회사용이에요. 이 자료 범위 밖.

➡️ 다음: [6.5 모니터링과 데이터 드리프트](05-모니터링.md)
