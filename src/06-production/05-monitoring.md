# 6.5 모니터링과 데이터 드리프트

## "배포한 모델이 이상해져요"

배포가 끝났다고 일이 끝난 게 아니에요. 진짜 일은 그 다음부터 시작돼요. 다음 같은 문제들이 일어납니다:

- 모델 정확도가 시간에 따라 떨어진다
- API 응답 시간이 느려진다
- 가끔 이상한 예측이 나온다
- 메모리가 점점 차오른다

이걸 일찍 감지하지 못하면 사용자가 먼저 알아채요. 그러면 큰일 나죠.

이 글에서는 **모델 운영 모니터링의 기본**을 다룹니다.

---

## 무엇을 모니터링해야 하나요?

크게 4가지예요.

### 1. 인프라 메트릭 (시스템 상태)

- CPU 사용률
- 메모리 사용률
- 디스크 사용률
- 네트워크 대역폭

웹 서비스 운영하시는 분에겐 익숙한 영역. 도구: Prometheus, Datadog, New Relic 등.

### 2. 서비스 메트릭 (API 상태)

- 요청 수 (RPS, requests per second)
- 응답 시간 (latency p50, p95, p99)
- 에러율
- 활성 연결 수

### 3. 모델 메트릭 (모델 상태)

- 예측 분포
- 입력 분포
- 모델 응답 시간
- 추론 메모리

### 4. 비즈니스 메트릭 (실제 영향)

- 모델 사용으로 인한 매출 변화
- 사용자 만족도
- 비즈니스 KPI

이 글은 **3번 (모델 메트릭)** 에 집중합니다.

---

## 모델 메트릭 1: 예측 분포 모니터링

학습 시 예측 분포와 운영 시 예측 분포가 비슷해야 해요.

### 학습 시

```python
import numpy as np
import matplotlib.pyplot as plt

# 학습 데이터에 대한 예측 분포 (베이스라인)
train_predictions = model.predict(X_train)
plt.hist(train_predictions, bins=50)
plt.title('학습 시 예측 분포')
plt.savefig('baseline_distribution.png')

# 통계 저장
baseline_stats = {
    'mean': train_predictions.mean(),
    'std': train_predictions.std(),
    'percentiles': {
        '25': np.percentile(train_predictions, 25),
        '50': np.percentile(train_predictions, 50),
        '75': np.percentile(train_predictions, 75),
    },
}

import json
with open('baseline_stats.json', 'w') as f:
    json.dump(baseline_stats, f, indent=2)
```

### 운영 시 (FastAPI 안에서)

```python
import json
from collections import deque
import numpy as np

# 최근 1000개 예측 저장
recent_predictions = deque(maxlen=1000)

# 베이스라인 통계 로드
with open('baseline_stats.json') as f:
    BASELINE = json.load(f)


@app.post("/predict")
def predict(features: HouseFeatures):
    prediction = model.predict(...)[0]
    
    # 예측 기록
    recent_predictions.append(prediction)
    
    # 주기적으로 분포 비교
    if len(recent_predictions) % 100 == 0:
        check_distribution()
    
    return {"prediction": prediction}


def check_distribution():
    recent = np.array(recent_predictions)
    
    # 평균과 표준편차 비교
    mean_diff = abs(recent.mean() - BASELINE['mean']) / BASELINE['mean']
    
    if mean_diff > 0.2:    # 20% 이상 다름
        logger.warning(
            f"Prediction distribution drift detected! "
            f"Recent mean: {recent.mean():.2f}, "
            f"Baseline mean: {BASELINE['mean']:.2f}"
        )
        # alert.send_slack("Drift detected!")    # 알람 보냄
```

---

## Data Drift (입력 데이터 변화)

운영 시 들어오는 입력이 학습 데이터와 달라지는 현상.

### 예시

```
학습 데이터:    20대 사용자 위주 (마케팅 결과)
                평균 나이 25
시간 흐른 후:   서비스 인기 → 모든 연령
                평균 나이 35

→ 모델은 20대용으로 학습됐는데, 다른 연령에는 잘 못 맞춤
```

### 감지 방법

각 입력 특성의 통계를 학습 시와 비교.

```python
class DriftDetector:
    def __init__(self, baseline_stats):
        self.baseline = baseline_stats
        self.recent_inputs = deque(maxlen=1000)
    
    def add_sample(self, features_dict):
        self.recent_inputs.append(features_dict)
    
    def check(self):
        if len(self.recent_inputs) < 100:
            return None
        
        drifts = {}
        
        for feature_name in self.baseline:
            recent_values = [s[feature_name] for s in self.recent_inputs]
            
            recent_mean = np.mean(recent_values)
            baseline_mean = self.baseline[feature_name]['mean']
            baseline_std = self.baseline[feature_name]['std']
            
            # z-score
            z = abs(recent_mean - baseline_mean) / baseline_std
            
            if z > 3:    # 3-sigma 벗어남
                drifts[feature_name] = {
                    'z_score': z,
                    'recent': recent_mean,
                    'baseline': baseline_mean,
                }
        
        return drifts


detector = DriftDetector(load_baseline_stats())

@app.post("/predict")
def predict(features: HouseFeatures):
    detector.add_sample(features.dict())
    
    # ... 예측 ...
    
    drifts = detector.check()
    if drifts:
        logger.warning(f"Drift detected: {drifts}")
```

### 더 정교한 도구

수학적으로 더 정확한 드리프트 감지:

- **KS 테스트** (Kolmogorov-Smirnov): 두 분포 비교
- **PSI** (Population Stability Index): 산업 표준
- **Wasserstein 거리**

```python
from scipy.stats import ks_2samp

def detect_drift_ks(baseline_data, recent_data, alpha=0.05):
    statistic, p_value = ks_2samp(baseline_data, recent_data)
    is_drift = p_value < alpha
    return is_drift, p_value
```

도구로는 **Evidently AI**, **WhyLabs** 등이 있어요.

---

## Concept Drift (관계 변화)

입력은 그대로인데 **입력과 출력의 관계가 바뀌는** 경우.

### 예시

```
2020년: 코로나 전 시대
        - 입력: 매장 위치, 날씨, 요일
        - 출력: 매장 매출

2021년: 코로나 시대 (관계 변화!)
        - 같은 입력
        - 다른 출력 (재택 증가, 외출 감소)

→ 옛 모델은 더 이상 정확하지 않음
```

### 감지 방법

**실제 결과(ground truth)** 와 비교해야 해요.

```python
@app.post("/predict")
def predict(features: HouseFeatures):
    prediction = model.predict(...)[0]
    
    # 예측 + 입력 저장 (DB나 파일)
    save_prediction(features, prediction)
    
    return {"prediction": prediction}


# 별도 잡으로 매일/매주 실행
def check_model_performance():
    # 지난 주 예측들
    predictions = get_recent_predictions(days=7)
    
    # 실제 결과 (수집 가능한 경우)
    actuals = get_actual_outcomes(days=7)
    
    # 매칭
    matched = match(predictions, actuals)
    
    rmse = np.sqrt(np.mean((matched['pred'] - matched['actual'])**2))
    
    if rmse > BASELINE_RMSE * 1.5:
        alert("Model degradation!")
```

**문제:** 실제 결과를 얻기 어려운 경우가 많아요.
- 사용자가 클릭 안 함 (CTR 모델)
- 결과가 몇 달 후에 나옴 (장기 예측)

이런 경우엔 입력 드리프트로 간접 감지하거나, 사람이 직접 라벨링.

---

## 알람 설정

### 무엇에 알람을 걸어야 하나?

- **응답 시간 p95 > 500ms**
- **에러율 > 1%**
- **드리프트 z-score > 3**
- **메모리 > 80%**
- **모델 성능 하락 (가능하면)**

### 누구한테 알람을?

- 즉각 대응 필요 → Slack DM
- 보통 → Slack 채널
- 정기 보고 → 이메일

### 알람 피로 (Alert Fatigue)

⚠️ 알람을 너무 많이 걸면 사람이 무시하기 시작합니다. 진짜 중요한 것만 알람으로.

---

## 로그와 메트릭

### 구조화 로그

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        return json.dumps(log_data)


# 사용
logger.info(
    "Prediction made",
    extra={
        'features': features_dict,
        'prediction': prediction,
        'model_version': '1.0.0',
        'duration_ms': duration_ms,
    },
)
```

JSON 로그는 검색과 집계가 쉬워요. ELK 스택, Grafana Loki 같은 도구로 분석.

### 메트릭 수집 (Prometheus 예시)

```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter(
    'ml_predictions_total',
    'Total predictions made',
    ['model_version'],
)

prediction_duration = Histogram(
    'ml_prediction_duration_seconds',
    'Prediction duration',
)


@app.post("/predict")
def predict(features: HouseFeatures):
    with prediction_duration.time():
        prediction = model.predict(...)
    
    prediction_counter.labels(model_version='1.0.0').inc()
    return {"prediction": prediction}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type='text/plain')
```

이걸 Prometheus가 주기적으로 수집해서 Grafana로 시각화.

---

## 대시보드 예시

좋은 ML 대시보드에는 다음이 있어야 해요.

```
┌─────────────────────────────────────────────────────┐
│  ML Model Dashboard - boston_house v1.0.0            │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ▸ 시스템                                              │
│    [응답시간 그래프]   p50: 23ms  p95: 87ms  p99: 234ms │
│    [요청 수]         142 RPS                          │
│    [에러율]          0.02%                            │
│                                                       │
│  ▸ 모델                                                │
│    [예측 분포]      [평균 추이]                        │
│    [입력 드리프트]   z-score 1.2 (정상)                │
│                                                       │
│  ▸ 비즈니스                                            │
│    [모델 사용 매출]                                    │
│    [모델 미사용 대비]                                  │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 모니터링 체크리스트

```
[ ] 헬스 체크 엔드포인트 (/health)
[ ] 로그 (구조화 JSON 권장)
[ ] 메트릭 수집 (Prometheus 등)
[ ] 응답 시간 측정 (p95, p99)
[ ] 에러율 모니터링
[ ] 입력 데이터 분포 추적
[ ] 예측 분포 추적
[ ] 데이터 드리프트 감지
[ ] (가능하면) 모델 성능 추적 (실제 결과 비교)
[ ] 알람 설정 (정말 중요한 것만)
[ ] 대시보드 (Grafana 등)
[ ] 정기 검토 (주간/월간)
```

---

## 도구 정리

| 카테고리 | 도구 |
|---------|------|
| **메트릭** | Prometheus, Datadog, New Relic |
| **로그** | ELK Stack, Loki, Splunk |
| **시각화** | Grafana, Kibana |
| **드리프트 감지** | Evidently AI, WhyLabs, Arize |
| **알람** | PagerDuty, Opsgenie, Slack 통합 |

---

## 정리

운영 후에 보세요:

1. **시스템**: CPU, 메모리, 디스크
2. **API**: 응답 시간, 에러율, 처리량
3. **모델**: 예측 분포, 입력 분포, 드리프트
4. **비즈니스**: 매출, 사용자 만족도

**한 줄 요약: "배포는 끝이 아니라 시작."**

➡️ 다음: [6.6 MLOps의 큰 그림](06-MLOps.md)
