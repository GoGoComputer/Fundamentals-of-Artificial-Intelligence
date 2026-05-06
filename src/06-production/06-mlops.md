# 6.6 MLOps의 큰 그림

## DevOps의 ML 버전

DevOps라는 단어 들어보셨죠? 소프트웨어 개발(Dev)과 운영(Ops)을 합친 말이에요. **개발과 운영의 경계를 자동화로 허무는** 사상.

**MLOps**는 ML에 그 사상을 적용한 거예요. ML 모델을:
- **자동으로** 학습하고
- **자동으로** 평가하고
- **자동으로** 배포하고
- **자동으로** 모니터링

이렇게 **사람의 손이 안 가는 사이클**을 만드는 게 MLOps의 목표입니다.

---

## ML 사이클의 큰 그림

```
       ┌────────────────────────────────────────┐
       │                                         │
       ↓                                         │
   [1] 데이터 수집                                │
       ↓                                         │
   [2] 데이터 검증                                │
       ↓                                         │
   [3] 특성 엔지니어링                             │
       ↓                                         │
   [4] 모델 학습                                  │
       ↓                                         │
   [5] 모델 평가                                  │
       ↓                                         │
   [6] 모델 검증                                  │
       ↓                                         │
   [7] 배포                                      │
       ↓                                         │
   [8] 모니터링                                  │
       ↓                                         │
   [9] 드리프트 감지 ─→ (조건 만족) ───→ [1]로
                                                 │
       └─────────────────────────────────────────┘
```

이 사이클이 **자동으로** 돌아가게 하는 게 MLOps예요.

---

## MLOps 성숙도

Google이 정의한 MLOps 성숙도가 유명해요. Level 0, 1, 2가 있어요.

### Level 0: 수동 (대부분의 회사)

```
데이터 사이언티스트가 노트북에서 학습
   ↓
모델 파일을 엔지니어에게 전달
   ↓
엔지니어가 수동으로 배포
   ↓
새 모델 필요할 때마다 처음부터
```

대부분의 회사가 여기에 있어요. 작은 프로젝트엔 충분.

### Level 1: ML 파이프라인 자동화

```
데이터가 들어오면 자동으로:
- 학습 → 평가 → 검증 → 배포

다만 파이프라인 자체는 사람이 만들고 유지
```

### Level 2: 완전 자동화 (CI/CD/CT)

```
파이프라인의 변경도 자동:
- 코드 푸시 → 자동 학습 → 자동 평가 → 자동 배포
- 모델이 자동으로 재학습 (CT: Continuous Training)
```

큰 회사들 (Google, Netflix, Uber)이 여기에 도달.

---

## 자동화의 단계별 도입

회사에서 MLOps를 도입하실 땐 한 번에 모든 걸 할 수 없어요. 점진적으로:

### 1단계: 실험 추적

가장 먼저 시도하기 좋아요.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("epochs", 10)
    
    model = train(...)
    
    mlflow.log_metric("test_accuracy", 0.945)
    mlflow.sklearn.log_model(model, "model")
```

**이득**: "어떤 실험이 어떤 결과였지?" 가 명확해짐. 손으로 엑셀에 적던 일이 자동.

### 2단계: 모델 레지스트리

학습된 모델을 중앙에서 관리.

```python
# 모델 등록
mlflow.register_model("runs:/abc123/model", "boston_predictor")

# 모델 가져오기
model = mlflow.pyfunc.load_model(
    "models:/boston_predictor/Production"
)
```

스테이지 (Staging, Production, Archived)로 모델 라이프사이클 관리.

### 3단계: 자동 학습 파이프라인

데이터가 새로 들어오면 자동으로 학습.

```python
# Airflow DAG 예시
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('weekly_retraining', schedule_interval='@weekly')

extract_data = PythonOperator(
    task_id='extract_data',
    python_callable=extract_recent_data,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train,
    dag=dag,
)

evaluate = PythonOperator(
    task_id='evaluate',
    python_callable=evaluate_model,
    dag=dag,
)

deploy = PythonOperator(
    task_id='deploy',
    python_callable=deploy_if_better,
    dag=dag,
)

extract_data >> train_model >> evaluate >> deploy
```

매주 자동으로 새 모델을 학습하고, 더 좋으면 배포.

### 4단계: A/B 테스트

새 모델 vs 기존 모델 비교.

```
Traffic
   ↓
   ┌─→ Model A (90%)  ─→ 메트릭 수집
   │
   └─→ Model B (10%)  ─→ 메트릭 수집

   비교 → B가 더 좋으면 점진적으로 100%로
```

도구: Optimizely, LaunchDarkly, 또는 자체 구현.

### 5단계: 완전 자동화

위 모든 게 사람 개입 없이 돌아감.

```
새 데이터
   ↓
자동 학습 + 평가 + 검증
   ↓
A/B 테스트
   ↓
승자 자동 배포
   ↓
모니터링 → 드리프트 감지 → 처음으로
```

---

## 도구 추천 (2026년 기준)

### 실험 추적 / 모델 레지스트리
- **MLflow** ⭐ 가장 인기, 무료
- **Weights & Biases** ⭐ 사용성 최고, 무료 시작
- **Neptune.ai**
- **Comet ML**

### 워크플로우 / 파이프라인
- **Airflow** ⭐ 표준, 광범위
- **Prefect** 더 현대적
- **Dagster** 데이터 중심
- **Kubeflow** Kubernetes 기반

### 모델 서빙
- **BentoML** ⭐ 쉬움, 빠름
- **Ray Serve** 분산
- **TorchServe** PyTorch 전용
- **Triton** NVIDIA, GPU 최적

### 특성 저장소 (Feature Store)
- **Feast** ⭐ 오픈소스
- **Tecton**

### 종합 플랫폼
- **AWS SageMaker** ⭐ 가장 인기
- **GCP Vertex AI**
- **Azure ML**
- **Databricks**

### 모니터링
- **Evidently AI** ⭐ ML 특화
- **Arize**
- **WhyLabs**
- **Fiddler AI**

---

## 작은 회사 / 개인 프로젝트의 MLOps

큰 회사처럼 다 갖추긴 어려워요. **작게 시작하세요.**

### 최소 구성

```
1. Git: 코드 + 노트북 버전 관리
2. MLflow: 실험 추적 (무료)
3. FastAPI + Docker: 모델 서빙
4. 클라우드 (Cloud Run 또는 App Runner): 배포
5. 간단한 로깅: print 또는 stdlib logging
```

이 5가지로 충분해요. 큰 도구들은 필요할 때 도입.

---

## MLOps의 핵심 원칙

### 1. 자동화 > 매뉴얼
사람이 손으로 할 때마다 실수가 생겨요. 한 번 자동화하면 영원히 안전.

### 2. 재현 가능성
같은 입력 → 같은 결과. 시드, 환경, 데이터 다 고정.

### 3. 버전 관리
모든 것에 버전. 코드, 데이터, 모델, 환경.

### 4. 모니터링
보지 않는 건 관리할 수 없어요.

### 5. 점진적 배포
A/B 테스트, 카나리 배포로 위험 최소화.

### 6. 빠른 피드백
실패가 일찍 보이면 일찍 고침.

---

## ML 엔지니어 vs 데이터 사이언티스트

회사에서 자주 보이는 두 직군. 차이를 짧게.

| | 데이터 사이언티스트 | ML 엔지니어 |
|---|---|---|
| 주력 | 모델링, 분석 | 인프라, 배포, 운영 |
| 도구 | Jupyter, sklearn, pandas | Docker, K8s, Airflow |
| 결과물 | 노트북, 리포트 | API, 파이프라인 |
| KPI | 모델 정확도 | 시스템 안정성 |

**MLOps**는 두 직군의 협업 영역이에요. 데이터 사이언티스트가 만든 모델을 ML 엔지니어가 운영 가능하게 만드는 다리.

---

## 더 깊이 가시려면

이 자료에서는 큰 그림만 다뤘어요. 실제로 MLOps 엔지니어가 되시려면 다음을 배우셔야 해요.

### 필수
- Linux 기본
- Git, GitHub Actions
- Docker, docker-compose
- Python (당연)
- 기본 SQL
- 기본 클라우드 (AWS 또는 GCP)

### 선택 (회사에 따라)
- Kubernetes
- Terraform (인프라 코드)
- 모니터링 도구 (Prometheus, Grafana)
- CI/CD (GitHub Actions, GitLab CI, Jenkins)
- 분산 컴퓨팅 (Spark, Ray, Dask)

자료:
- 책: "Designing Machine Learning Systems" (Chip Huyen) — 가장 좋은 입문서
- 책: "Machine Learning Engineering" (Andriy Burkov)
- 강의: Coursera "MLOps Specialization" (Andrew Ng)
- 사이트: [madewithml.com](https://madewithml.com)

---

## 정리

```
MLOps = ML + DevOps + 자동화

핵심 원칙:
- 자동화 (반복은 자동화)
- 재현성 (모든 게 결정적)
- 버전 관리 (모든 것에 버전)
- 모니터링 (보고 알람)
- 점진적 배포 (A/B 테스트)

도구는 다양하지만 사상은 같음.
작은 것부터 시작하세요.
```

---

## 6장을 마치며 — 그리고 자료 전체를 마치며

여기까지 오신 모든 분, 진심으로 축하드립니다.

이 자료에서 배운 것을 한 번에 정리하면:

```
[기초]
0장: 왜 AI인가 + 환경 설정
1장: 파이썬 (필요한 만큼만)

[머신러닝]
2장: 분류 (SVM, RF, 로지스틱, KNN)
3장: 회귀 (선형, 정규화, 트리 기반)

[딥러닝]
4장: 분류 (PyTorch, 신경망)
5장: 회귀 (과적합 + 3가지 무기)

[프로덕션]
6장: 패키징 → 서빙 → Docker → 모니터링 → MLOps
```

이 정도면 **현업 ML 엔지니어 입문 수준**입니다. 다음으로:

### 더 배우고 싶으시면

[부록의 더 나아가기](../../부록/더-나아가기.md)를 보세요. 다음 학습을 안내합니다.

### 본인 프로젝트를 시작하세요

가장 좋은 학습은 **본인의 데이터로 본인의 문제를 풀어 보기**예요. 캐글(Kaggle)에 도전하시거나, 본인 일상의 문제(예: 영화 추천, 일기 분류)에 적용해 보세요.

### 막히면 돌아오세요

이 자료는 책처럼 한 번 읽고 끝나는 게 아니에요. **참고서**입니다. 회사에서 일하시면서 막히는 게 있으면 해당 챕터로 돌아와 다시 보세요.

---

**좋은 길 가십시오. 다 잘 되실 겁니다.**

➡️ [부록: 용어 사전](../../부록/용어-사전.md)
➡️ [부록: 자주 묻는 질문](../../부록/자주-묻는-질문.md)
➡️ [부록: 더 나아가기](../../부록/더-나아가기.md)
