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


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **6.4 Docker로 패키징** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **6.4 Docker로 패키징**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

---

## 자주 하는 오해와 바로잡기

### 1) "이론만 이해하면 실습은 저절로 된다"
아닙니다. 이론 이해와 코드 실행 능력은 별개의 근육입니다. 둘을 같이 써야 합니다.

### 2) "예제 코드가 돌아갔으니 완전히 이해했다"
동작 확인은 시작일 뿐입니다. 파라미터를 바꿨을 때 결과가 어떻게 달라지는지까지 확인해야 이해가 완성됩니다.

### 3) "좋은 모델은 항상 하나다"
데이터 특성과 제약(시간, 메모리, 응답 속도)에 따라 최선의 선택은 달라집니다.

### 4) "지표 숫자만 높으면 끝이다"
지표는 해석이 필요합니다. 어떤 데이터에서, 어떤 분포에서, 어떤 기준으로 얻은 점수인지 함께 보아야 합니다.

### 5) "지금은 초반이니까 배포는 나중에 생각해도 된다"
초반부터 재현성과 저장 구조를 습관화해야 나중에 배포 단계에서 무너지지 않습니다.

---

## 실전 연결 시나리오

아래는 이 단원 내용을 실제 업무로 연결할 때 자주 쓰는 흐름입니다.

1. 문제를 한 문장으로 정의한다.  
예: "6.4 Docker로 패키징 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

2. 성공 기준을 지표로 정한다.  
예: 정확도/RMSE/F1/지연시간 같은 운영 가능한 숫자로 정의.

3. 가장 단순한 베이스라인을 만든다.  
복잡한 모델보다 먼저, 빠르게 재현 가능한 기본 모델을 세웁니다.

4. 한 번에 하나씩 개선한다.  
전처리, 모델, 튜닝, 임계값, 배치 크기 등은 동시에 바꾸지 않고 순차적으로 바꿉니다.

5. 결과를 로그와 함께 저장한다.  
실험 이름, 파라미터, 점수, 데이터 버전을 함께 남겨야 재현이 됩니다.

6. 실패 케이스를 분석한다.  
점수 평균보다 오분류/고오차 샘플을 분석할 때 개선 아이디어가 나옵니다.

7. 배포 가능 형태로 정리한다.  
모델 파일, 메타데이터, 입력 스키마, 테스트 스크립트까지 묶습니다.

---

## 복습 체크리스트

다음 질문에 답할 수 있으면 이 단원을 실전 수준으로 이해한 것입니다.

- 이 단원의 핵심 개념을 중학생에게 설명하듯 말할 수 있는가?
- 코드의 각 블록이 왜 필요한지 한 줄씩 설명할 수 있는가?
- 이 단원에서 가장 자주 틀리는 지점을 알고 있는가?
- 지표를 볼 때 어떤 함정이 있는지 알고 있는가?
- 베이스라인과 개선 모델의 차이를 표로 정리할 수 있는가?
- 재현 가능하게 실험을 저장하는 습관이 있는가?
- 실패 사례를 보고 다음 실험 가설을 만들 수 있는가?
- 이 단원 내용을 다음 장과 어떻게 연결할지 설명할 수 있는가?

---

## 확장 연습 과제

### 과제 A: 파라미터 감각 만들기
현재 예제에서 핵심 파라미터 1~2개만 바꿔 5회 실험하고, 결과를 표로 정리해 보세요.

### 과제 B: 실패 사례 분석
오분류 또는 오차가 큰 샘플 20개를 모아 공통 패턴을 찾아 보세요.

### 과제 C: 설명 가능 리포트 작성
"무엇을 했고, 왜 했고, 결과가 어땠고, 다음엔 무엇을 할지"를 1페이지로 정리해 보세요.

이 3개를 직접 해 보면 **6.4 Docker로 패키징**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

6.4 Docker로 패키징는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
