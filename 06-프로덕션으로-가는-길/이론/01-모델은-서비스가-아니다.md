# 6.1 모델 ≠ 서비스

## "모델은 다 만들었어요. 이제 어떻게 서비스로 만들어요?"

이 질문이 회사에서 ML을 시작하는 모든 사람의 첫 막힘 지점이에요. 답을 한 줄로 하면:

> **노트북 안의 모델과 사용자에게 도달하는 모델 사이엔 거대한 강이 있다.**

이 강을 건너는 게 이 장의 주제입니다.

---

## 한 비유

여러분이 김치를 정말 잘 담그신다고 해 봐요. 본인 부엌에서 만든 김치는 환상이에요. 친구들도 다 좋아해요.

그런데 이걸 **"전국에 매일 1만 가구에 배송하는 사업"** 으로 만들려면 어떻게 해야 할까요?

- 큰 공장이 필요해요
- 매일 정해진 시간에 만들어야 해요
- 품질이 일관돼야 해요
- 배송 시스템이 있어야 해요
- 누가 하나 사 먹고 식중독에 걸리면 큰일 나요
- 김장 한 번에 5천 포기 만들 수 있어야 해요
- ...

집에서 김치 잘 담그는 것과 김치 사업하는 건 **완전히 다른 일**이에요.

ML도 똑같습니다. 노트북에서 잘 동작하는 모델과, 1만 명이 동시에 쓰는 서비스는 **완전히 다른 일**이에요.

---

## 노트북 vs 프로덕션 — 무엇이 다른가

| 측면 | 노트북 | 프로덕션 |
|-----|--------|---------|
| **사용자** | 본인 한 명 | 1~수백만 명 |
| **속도 요구** | 좀 느려도 됨 | 100ms 이내 응답 |
| **안정성** | 죽으면 다시 실행 | 24/7 무중단 |
| **데이터** | 깨끗한 학습 데이터 | 지저분한 실제 데이터 |
| **에러 처리** | 죽으면 그만 | 우아하게 처리 |
| **모니터링** | print 문 | 대시보드, 알람 |
| **재학습** | 필요할 때 수동 | 자동화된 파이프라인 |

각 항목이 그냥 "조금 어려워지는" 게 아니라 **완전히 다른 도구와 다른 사고방식**이 필요해요.

---

## 잘 알려진 함정 5가지

이 장의 나머지 글들에서 자세히 다룰 함정들을 미리 보여드릴게요.

### 함정 1: "학습 때랑 추론 때 데이터가 달라요"

```python
# 학습 (예쁜 정규화된 데이터)
X = scaler.fit_transform(clean_data)
model.fit(X, y)

# 운영 (사용자가 보낸 새 데이터)
new_data = api_request_body['features']    # ← 그냥 들어옴
y_pred = model.predict(new_data)            # ← 정규화 안 함!
```

`scaler`를 같이 저장하지 않으면 운영에서 잘못된 예측이 나와요. → [6.2](02-패키징-저장.md)에서 다룸.

### 함정 2: "모델 응답이 너무 느려요"

학습은 한 번이지만 추론은 매 요청마다. **응답 속도가 사용자 경험을 좌우합니다.**

- 사용자는 100ms 이상 기다리면 답답해함
- 1초 이상이면 다른 페이지로 떠남

→ [6.3](03-FastAPI-서빙.md), [6.4](04-Docker.md)에서 다룸.

### 함정 3: "배포 후에 모델 성능이 떨어졌어요"

세상이 변하면 데이터도 변해요. 학습 때와 다른 데이터가 들어오면 모델 성능이 떨어집니다. (Concept Drift, Data Drift)

→ [6.5](05-모니터링.md)에서 다룸.

### 함정 4: "재학습은 어떻게 해요?"

매주 새로운 데이터가 쌓여요. 모델을 새로 학습시켜야 해요. 손으로 매번?

→ [6.6](06-MLOps.md)에서 다룸.

### 함정 5: "어떤 모델이 운영 중인지 모르겠어요"

모델이 여러 버전 있어요. 어느 게 어디 배포돼 있는지, 누가 학습했는지, 어떤 데이터로 했는지...

→ 모델 레지스트리, 메타데이터 관리. [6.6](06-MLOps.md)에서.

---

## 모델 출시 전 마지막 체크 5가지

회사에서 모델을 운영에 올리기 전, 다음 다섯 가지를 꼭 확인하세요.

### 1. 추론 속도 (Latency)

```python
import time

start = time.time()
prediction = model.predict(sample_input)
duration_ms = (time.time() - start) * 1000

print(f"단일 추론 시간: {duration_ms:.2f}ms")
```

목표: **100ms 이하** (실시간 서비스), **1000ms 이하** (배치 처리)

### 2. 처리량 (Throughput)

초당 몇 번 요청을 받을 수 있나?

```python
n_requests = 1000
start = time.time()
for _ in range(n_requests):
    model.predict(sample)
duration = time.time() - start

print(f"초당 처리량: {n_requests/duration:.0f} req/s")
```

### 3. 메모리 사용량

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"메모리: {mem_mb:.0f} MB")
```

큰 모델이 작은 서버에 안 올라갈 수 있어요.

### 4. 입력 검증

이상한 입력이 와도 죽지 않아야 해요.

```python
def predict_safe(features):
    # 검증
    if not isinstance(features, dict):
        return {"error": "Input must be dict"}
    
    if 'feature_1' not in features:
        return {"error": "Missing feature_1"}
    
    if not (0 <= features['feature_1'] <= 100):
        return {"error": "feature_1 out of range"}
    
    # 예측
    try:
        result = model.predict(features)
        return {"prediction": float(result)}
    except Exception as e:
        return {"error": str(e)}
```

### 5. 폴백 (Fallback)

모델이 실패했을 때 무엇을 답할까?

```python
def predict_with_fallback(features):
    try:
        return model.predict(features)
    except:
        return get_default_prediction()    # 평균값, 가장 흔한 값 등
```

---

## "이거 다 우리가 해야 하나요?"

좋은 소식 하나 드릴게요. **요즘은 많은 게 자동화돼 있어요.**

### 옵션 1: 클라우드 서비스 사용

- **AWS SageMaker** — 학습부터 배포까지 다 관리
- **GCP Vertex AI** — 구글의 ML 플랫폼
- **Azure ML** — 마이크로소프트
- **Hugging Face Inference Endpoints** — 사전학습 모델 위주

이 서비스들을 쓰면 **인프라의 90%가 해결**돼요. 모델 코드만 가지고 배포 가능.

### 옵션 2: 오픈소스 도구

- **MLflow** — 실험 추적 + 모델 레지스트리
- **BentoML** — 모델 서빙
- **Ray Serve** — 분산 모델 서빙
- **Kubeflow** — 쿠버네티스 기반 ML 파이프라인
- **Airflow / Prefect** — 워크플로우 자동화

### 옵션 3: 직접 만들기

가장 자유롭지만 가장 복잡. 큰 회사들은 이 길을 갑니다 (Google, Meta, Netflix 등).

---

## 이 장의 접근

이 장은 **옵션 3 (직접 만들기)** 의 가장 단순한 버전을 다룹니다. 이유:

1. 이걸 이해하면 다른 옵션들도 쉽게 이해돼요
2. 작은 프로젝트엔 이 정도면 충분
3. 어떤 도구든 결국 비슷한 것을 자동화할 뿐이에요

회사에서 진짜 일을 하실 땐 클라우드 서비스나 본인 회사 인프라를 쓰시게 될 거예요. 그때 "아, 이게 [이 장에서 본] 그거구나" 하고 이해하시면 됩니다.

---

## 정리

```
노트북에서 학습 끝난 모델
        ↓
   [Pipeline 저장]
        ↓
   [API 서버 (FastAPI)]
        ↓
   [Docker 패키징]
        ↓
   [배포 (서버, 클라우드)]
        ↓
   [모니터링]
        ↓
   [재학습 → 처음으로]
```

이 사이클이 **MLOps**예요. 다음 글부터 한 단계씩 봅시다.

➡️ 다음: [6.2 모델 패키징과 저장](02-패키징-저장.md)
