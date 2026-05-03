# 3.6 트리 기반 회귀

## 또 다른 패러다임

지금까지의 회귀 모델들(LinearRegression, Lasso, Ridge)은 **수식**으로 답을 만들었어요.

```
집값 = 4.5 × 방수 - 0.5 × 저소득층비율 + ...
```

**트리 기반 모델**은 완전히 다른 접근이에요. **스무고개**처럼 질문을 거치며 답을 좁혀 나갑니다.

```
방 개수 > 6?
├ 네: 저소득층 < 10%?
│     ├ 네: 평균 가격 = 35
│     └ 아니오: 평균 가격 = 25
└ 아니오: 학군 점수 > 15?
      ├ 네: 평균 가격 = 22
      └ 아니오: 평균 가격 = 15
```

이런 식으로 트리(나무) 구조의 결정 흐름을 만들어요. 그래서 **결정 트리(Decision Tree)** 라고 부릅니다.

---

## 1. 결정 트리 회귀 (Decision Tree)

### 코드

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 주요 옵션

- `max_depth`: 트리의 최대 깊이 (제한 안 두면 과적합)
- `min_samples_split`: 분할에 필요한 최소 샘플 수
- `min_samples_leaf`: 잎 노드의 최소 샘플 수

### 장점

- ✅ **이해하기 쉬움** — 트리를 그릴 수 있음
- ✅ **정규화 필요 없음**
- ✅ **비선형 관계 잘 잡음**

### 단점

- ❌ **단일 트리는 과적합**에 매우 취약
- ❌ **불안정** — 데이터 살짝 바뀌면 다른 트리

이 단점들을 해결한 게 **앙상블(ensemble)** 이에요. 여러 트리를 합치는 거죠.

---

## 2. Random Forest 회귀

여러 개의 결정 트리를 학습시키고, **그 평균을 답으로 합니다.** 분류에서 나온 거랑 같은 알고리즘인데, 답을 만드는 방식만 살짝 달라요.

- 분류: 트리들의 다수결
- 회귀: 트리들의 평균

### 코드

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,    # 트리 100개
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 왜 잘 동작하나요?

각 트리가 살짝씩 다르게 학습돼요(데이터의 부분집합으로). 그래서 **트리들이 각자 다른 실수를 합니다.** 100개의 트리가 모이면, 개별 실수는 평균에서 상쇄돼요.

> "한 명의 전문가보다 100명의 일반인의 평균이 더 정확하다" — 군중의 지혜

### 특성 중요도

```python
import pandas as pd

importances = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('중요도')
plt.title('Random Forest 특성 중요도')
plt.tight_layout()
plt.show()
```

이게 Random Forest의 또 다른 매력이에요. **어떤 변수가 중요했는지** 알려줘요. 선형 회귀의 가중치와는 다른 종류의 정보예요.

---

## 3. Gradient Boosting

이건 아주 강력한 알고리즘이에요. **이전 트리의 실수를 다음 트리가 고치도록** 순차적으로 학습합니다.

```
1단계: 첫 트리 학습 → 일부 예측은 틀림
2단계: 그 틀림을 메꾸는 두 번째 트리 학습
3단계: 또 남은 틀림을 메꾸는 세 번째 트리 학습
...
N단계: 모든 트리의 예측을 합쳐서 최종 답
```

이 과정을 **부스팅(boosting)** 이라고 해요.

### 코드

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,    # 각 트리의 영향력
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)
```

### Random Forest vs Gradient Boosting

| | Random Forest | Gradient Boosting |
|---|---|---|
| 학습 방식 | 병렬 (각자 독립) | 순차 (앞 트리를 보완) |
| 학습 속도 | 빠름 | 느림 |
| 예측 정확도 | 좋음 | 보통 더 좋음 |
| 과적합 위험 | 낮음 | 높음 (튜닝 중요) |
| 튜닝 난이도 | 쉬움 | 어려움 |

---

## 4. XGBoost — 캐글의 황제

`GradientBoostingRegressor`의 **개선된 버전**이에요. 속도와 정확도 모두 끌어올린 라이브러리. 캐글(Kaggle) 대회에서 가장 자주 우승하는 알고리즘이에요.

### 설치

Colab에는 기본으로 깔려 있어요. 본인 PC에서는:

```bash
pip install xgboost
```

### 코드

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

sklearn 인터페이스를 그대로 따르기 때문에 사용법이 익숙하실 거예요.

### 주요 옵션

- `n_estimators`: 트리 개수
- `learning_rate`: 학습률 (작을수록 천천히, 더 많은 트리 필요)
- `max_depth`: 트리 깊이
- `subsample`: 각 트리가 쓸 샘플 비율
- `colsample_bytree`: 각 트리가 쓸 특성 비율
- `reg_alpha`, `reg_lambda`: L1/L2 정규화

### 좋은 시작 옵션

```python
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,    # 검증 점수 안 좋아지면 일찍 멈춤
)

# 검증셋과 함께
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
```

---

## 5. LightGBM, CatBoost (참고)

XGBoost 같은 라이브러리들이 더 있어요.

| 라이브러리 | 특징 |
|------|------|
| **LightGBM** | XGBoost보다 빠름. 큰 데이터에서 유리 |
| **CatBoost** | 범주형 변수 자동 처리. 튜닝 덜 필요 |

**셋 다 거의 같은 인터페이스예요.** 하나 쓸 줄 알면 다 쓸 수 있어요.

```python
from lightgbm import LGBMRegressor
model = LGBMRegressor(...)

from catboost import CatBoostRegressor
model = CatBoostRegressor(...)
```

---

## 한꺼번에 비교

```python
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# XGBoost 있으면 추가
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False


# 데이터 준비 (Boston, 전 글 코드 재사용)
# X_train, X_test, y_train, y_test = ...


models = {
    "Ridge":             Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
if has_xgb:
    models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)


for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.time() - start

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name:25s}  RMSE: {rmse:.3f}  R²: {r2:.4f}  ({duration:.2f}s)")
```

```
Ridge                      RMSE: 4.928  R²: 0.6688  (0.01s)
Decision Tree              RMSE: 3.812  R²: 0.8016  (0.02s)
Random Forest              RMSE: 2.882  R²: 0.8866  (0.45s)
Gradient Boosting          RMSE: 2.798  R²: 0.8932  (0.21s)
XGBoost                    RMSE: 2.687  R²: 0.9012  (0.18s)
```

**XGBoost가 가장 정확하고, 속도도 빠른 걸 볼 수 있어요.** 그래서 캐글에서 사랑받습니다.

---

## 모델 선택 가이드 (회귀)

회사에서 회귀 문제를 만나면 이런 순서로 시도해 보세요.

```
1. 데이터 탐색
   ↓
2. 일단 Linear/Ridge 한 번 돌려서 베이스라인
   ↓
3. Random Forest로 개선되는지 확인
   ↓
4. 만족스러우면 → 끝
   더 필요하면 → XGBoost
   ↓
5. 그래도 안 되면 → 딥러닝 (5장)
   또는 → 데이터/특성 다시 보기
```

**황금 법칙: 단순한 것부터.** Linear 베이스라인보다 50% 더 잘 하지 못하는 복잡한 모델은 의미가 없어요.

---

## 트리 모델은 정규화 필요 없음

트리 모델의 큰 장점 중 하나예요. 변수의 스케일이 달라도 자동으로 알아서 나눕니다.

```python
# Ridge: 정규화 필요
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge()),
])

# RandomForest: 정규화 안 해도 됨
model = RandomForestRegressor()    # 그냥 X_train 넣으면 됨
```

이것만 해도 데이터 전처리에 들어가는 코드가 절반으로 줄어요.

---

## 정리

| 모델 | 언제? |
|------|------|
| Linear / Ridge | 베이스라인, 해석 필요 |
| Lasso | 특성이 많고 일부만 중요할 때 |
| Decision Tree | 결과를 시각화/설명할 때 |
| Random Forest | 거의 항상 (안전한 선택) |
| Gradient Boosting | 정확도 더 필요할 때 |
| **XGBoost / LightGBM** | **정확도 최우선** (현업 표준) |

코드 패턴은 모두 동일해요. **`fit`, `predict`** — 이 두 줄만 알면 다 쓸 수 있어요.

➡️ 다음: [3.7 현업 체크리스트](07-현업-체크리스트.md)
