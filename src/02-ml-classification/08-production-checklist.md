# 2.8 현업 체크리스트

> 이 글은 회사에서 분류 모델을 진지하게 만드시는 분들을 위한 글입니다.
> 학습용으로만 쓰실 분도 한 번쯤은 읽어보시면 좋아요.
> "이런 게 회사에서는 추가로 신경 써야 하는구나" 정도로요.

---

## 회사 ML 코드와 학습용 코드의 차이

학습용 코드:
```python
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

회사 코드 (간략화):
```python
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_path: str):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(C=1.0, kernel='rbf', random_state=42)),
    ])
    
    logger.info(f"Training on {X_train.shape[0]} samples")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")
    
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return pipeline
```

뭐가 다를까요?

1. **Pipeline 사용** — 전처리와 모델을 한 묶음으로
2. **로깅** — 무슨 일이 언제 있었는지 기록
3. **함수화** — 재사용 가능한 단위로
4. **모델 저장** — 매번 다시 학습 안 하게
5. **타입 힌트** — `model_path: str`

이런 작은 차이들이 모이면 운영 가능한 시스템이 됩니다. 하나씩 보겠습니다.

---

## 체크리스트 1: 데이터 검증

학습 데이터에 다음을 검증하셨나요?

### ✅ 결측값 (NaN) 확인
```python
import pandas as pd

# 결측값 개수
print(df.isnull().sum())

# 결측값 있는 행 비율
print(df.isnull().any(axis=1).mean())
```

결측값이 있으면 어떻게 처리할지 결정하셔야 해요.
- 삭제 (`df.dropna()`)
- 채우기 (`df.fillna(0)` 또는 평균/중앙값)
- 모델 안에서 처리 (XGBoost 등은 결측값을 자동 처리)

### ✅ 이상치 (Outlier) 확인
```python
# 박스플롯으로 시각화
import seaborn as sns
sns.boxplot(data=df)

# 또는 IQR 기준
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"이상치 비율: {outliers.mean():.2%}")
```

### ✅ 클래스 분포 확인
```python
print(y.value_counts(normalize=True))
```

너무 불균형하면(예: 99:1) 별도 처리 필요.

### ✅ 중복 확인
```python
print(f"중복 행: {df.duplicated().sum()}")
```

### ✅ 데이터 누수 확인
- test 데이터의 정보가 train으로 흘러갔는가?
- 미래 정보가 과거 학습에 들어갔는가? (시계열에서 매우 중요)
- 같은 사용자/세션이 train과 test에 모두 있는가?

---

## 체크리스트 2: 코드 구조

### ✅ 함수로 분리
```python
def load_data(path: str):
    ...

def preprocess(df: pd.DataFrame):
    ...

def train(X, y):
    ...

def evaluate(model, X_test, y_test):
    ...
```

각 단계가 독립적이어야 테스트 가능해요.

### ✅ Pipeline 사용

전처리 + 모델을 하나의 객체로 묶어 두면, 새 데이터에 자동으로 같은 전처리가 적용돼요.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf')),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)   # 자동으로 scaler 적용됨
```

서비스에 배포할 때 정말 중요해요. 모델만 저장하면 전처리 일치를 사람이 챙겨야 하지만, Pipeline을 저장하면 다 같이 저장돼요.

### ✅ 타입 힌트와 문서화

```python
from typing import Tuple
import numpy as np


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
) -> Pipeline:
    """랜덤 포레스트 분류기를 학습한다.
    
    인자:
        X_train: 학습 데이터 (n_samples, n_features)
        y_train: 라벨 (n_samples,)
        n_estimators: 트리 개수
    
    반환:
        학습된 sklearn Pipeline
    """
    ...
```

### ✅ 설정값 분리

하이퍼파라미터를 코드 안에 박아두지 말고 설정 파일이나 변수로:

```python
# config.py 또는 dict
CONFIG = {
    'model': {
        'type': 'random_forest',
        'n_estimators': 200,
        'max_depth': 20,
    },
    'training': {
        'test_size': 0.2,
        'random_state': 42,
    },
    'paths': {
        'data': 'data/raw.csv',
        'model': 'models/rf_v1.pkl',
    },
}
```

---

## 체크리스트 3: 평가

### ✅ 다양한 지표 봤는가
- 정확도뿐만 아니라 precision, recall, F1
- 불균형 데이터면 ROC-AUC, PR-AUC도

### ✅ 클래스별로 봤는가
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

전체 평균만 보면 특정 클래스에서 망하는 걸 못 봐요.

### ✅ 혼동 행렬을 시각적으로 봤는가
어느 클래스 → 어느 클래스로 자주 잘못 가는지 패턴 파악.

### ✅ 학습 데이터에서의 성능도 봤는가
```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train: {train_score:.4f}")
print(f"Test:  {test_score:.4f}")
print(f"Gap:   {train_score - test_score:.4f}")
```

- gap이 크다 → 과적합 (모델이 너무 복잡하거나, 데이터 부족)
- gap이 작은데 둘 다 낮다 → 과소적합 (모델이 너무 단순하거나, 특성 부족)

### ✅ 실제 운영 데이터와 비슷한 환경에서 평가했는가

예: 시계열 데이터인데 무작위로 train/test 나누면 안 됨. 시간 순으로 나눠야 함.

```python
# 잘못된 방법 (시계열)
train_test_split(X, y, shuffle=True)

# 올바른 방법
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

## 체크리스트 4: 재현성

### ✅ 모든 random_state 고정
```python
RANDOM_STATE = 42

train_test_split(..., random_state=RANDOM_STATE)
RandomForestClassifier(random_state=RANDOM_STATE)
RandomizedSearchCV(..., random_state=RANDOM_STATE)
```

### ✅ 라이브러리 버전 고정
```bash
# requirements.txt 또는 pyproject.toml
scikit-learn==1.5.0
numpy==1.26.0
pandas==2.2.0
```

또는 `uv lock`을 사용하세요.

### ✅ 데이터 버전 관리
- 어떤 시점의 데이터로 학습했는가?
- 데이터의 해시(checksum) 기록
- 또는 [DVC](https://dvc.org) 같은 도구 사용

### ✅ 실험 추적 도구
- [MLflow](https://mlflow.org)
- [Weights & Biases](https://wandb.ai)
- [Neptune](https://neptune.ai)

이 도구들은 "어떤 하이퍼파라미터로 어떤 데이터로 학습해서 어떤 결과가 나왔는지"를 자동으로 기록해 줘요. 회사에서는 거의 표준이에요.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=20)
    model.fit(X_train, y_train)
    
    score = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", score)
    
    mlflow.sklearn.log_model(model, "model")
```

---

## 체크리스트 5: 모델 저장과 버전 관리

### ✅ 모델 저장 방법

```python
# joblib (큰 모델에 좋음)
import joblib
joblib.dump(pipeline, 'model_v1.pkl')

# 불러오기
pipeline = joblib.load('model_v1.pkl')
```

### ✅ 메타데이터도 같이

모델 파일만 있으면 안 돼요. 다음도 같이:

```python
metadata = {
    'model_version': '1.0.0',
    'training_date': '2026-05-03',
    'training_data_hash': 'abc123...',
    'features': ['feature_1', 'feature_2', ...],
    'metrics': {
        'accuracy': 0.945,
        'f1_weighted': 0.943,
    },
    'hyperparameters': {...},
    'sklearn_version': sklearn.__version__,
}

import json
with open('model_v1.metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### ✅ 모델 디렉토리 구조 예시

```
models/
├── classifier_v1.0.0/
│   ├── model.pkl
│   ├── metadata.json
│   ├── feature_names.json
│   └── README.md
└── classifier_v1.1.0/
    └── ...
```

---

## 체크리스트 6: 운영 단계

### ✅ 입력 검증

배포 후엔 어떤 데이터가 들어올지 몰라요. 학습 때와 다른 형태가 들어올 수 있어요.

```python
def validate_input(features):
    if not isinstance(features, dict):
        raise ValueError("Input must be a dict")
    
    required = ['age', 'income', 'education']
    missing = [f for f in required if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    if not (0 <= features['age'] <= 150):
        raise ValueError(f"Age out of range: {features['age']}")
```

### ✅ 모델 응답 시간 측정

```python
import time

start = time.time()
prediction = model.predict(X_new)
duration = time.time() - start

logger.info(f"Prediction took {duration*1000:.2f}ms")
```

응답 시간이 너무 길면(>100ms) 모델을 더 가벼운 걸로 바꾸거나, ONNX 같은 추론 엔진을 고려해야 해요.

### ✅ 예측 결과 로깅

```python
logger.info(f"Input: {features}, Prediction: {pred}, Confidence: {conf}")
```

나중에 "왜 이런 예측이 나왔지?"를 디버깅할 때 정말 필요해요.

### ✅ 모니터링

배포 후에도 모델은 계속 봐야 해요.
- **모델 성능**: 정확도가 시간에 따라 떨어지진 않는가? (concept drift)
- **데이터 분포**: 입력 데이터가 학습 때와 비슷한가? (data drift)
- **응답 시간**: 너무 느려지진 않는가?
- **에러율**: 예측이 실패하는 비율

이건 6장(프로덕션)에서 자세히 다룹니다.

---

## 체크리스트 7: 윤리와 공정성

진지하게 모델을 만드시면 이것도 꼭 봐야 해요.

### ✅ 편향(Bias) 확인

학습 데이터에 편향이 있으면, 모델은 그 편향을 그대로 배워요.

- 채용 모델이 특정 성별을 차별하지 않는가?
- 대출 심사가 특정 지역을 차별하지 않는가?
- 의료 진단이 특정 인종에 정확도가 떨어지지 않는가?

```python
# 예: 성별별 성능 비교
from sklearn.metrics import accuracy_score

male_idx = X_test['gender'] == 'M'
female_idx = X_test['gender'] == 'F'

acc_male = accuracy_score(y_test[male_idx], y_pred[male_idx])
acc_female = accuracy_score(y_test[female_idx], y_pred[female_idx])

print(f"남성: {acc_male:.4f}, 여성: {acc_female:.4f}")
```

차이가 크다면 데이터 수집부터 다시 봐야 해요.

### ✅ 설명 가능성

일부 분야(의료, 금융, 채용)는 **법적으로** "왜 이런 예측을 했는가"를 설명해야 해요.

도구:
- **SHAP** (`shap` 패키지) — 가장 인기
- **LIME** — 로컬 설명
- 트리 기반 모델의 `feature_importances_`

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:10])

shap.summary_plot(shap_values, X_test[:10])
```

---

## 체크리스트: 한 페이지 요약

학습용 → 회사용으로 가는 길

```
[ ] 데이터 검증 (NaN, 이상치, 클래스 분포, 중복, 누수)
[ ] 함수로 분리, Pipeline 사용
[ ] 타입 힌트, docstring
[ ] 설정값을 코드에서 분리
[ ] 다양한 평가 지표 사용
[ ] train/test 성능 차이 확인 (과적합)
[ ] 클래스별, 그룹별 성능 확인
[ ] 모든 random_state 고정
[ ] 라이브러리 버전 고정
[ ] 실험 추적 (MLflow 등)
[ ] 모델 + 메타데이터 함께 저장
[ ] 입력 검증
[ ] 응답 시간 측정 + 로깅
[ ] 모니터링 계획
[ ] 편향 확인
[ ] 설명 가능성 도구
```

이 체크리스트 다 통과하시면, **회사 면접에서 "ML 엔지니어" 라고 자신 있게 말할 수 있게 됩니다.** 진짜로요.

---

## 2장을 마치며

분류 챕터, 끝났습니다! 정리해 보면:

- ✅ 분류가 무엇이고 어떤 문제에 쓰이는지
- ✅ MNIST 같은 데이터를 다루는 법
- ✅ 데이터 분할과 정규화
- ✅ SVM부터 Random Forest까지 4가지 모델
- ✅ 정확도, F1, ROC-AUC 같은 평가 지표
- ✅ Grid/Random Search로 하이퍼파라미터 튜닝
- ✅ 회사에서 ML을 할 때 신경 써야 할 것들

다음 장은 **회귀**예요. 분류와 비슷하지만 답이 카테고리가 아닌 **숫자**라는 점만 달라요. 코드 패턴은 거의 같으니 금방 익히실 거예요.

➡️ **[3장. 머신러닝 — 회귀](../../03-머신러닝-회귀/)**
