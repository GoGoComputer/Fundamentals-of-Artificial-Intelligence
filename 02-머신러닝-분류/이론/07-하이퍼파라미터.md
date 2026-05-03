# 2.7 하이퍼파라미터 튜닝

## "이 모델, 더 좋게 만들 수 있어요?"

지금까지 우리는 모델의 옵션들(`C=1.0`, `n_estimators=100`)을 그냥 기본값으로 썼어요. 그런데 이 옵션들을 잘 조절하면 정확도가 **2~5%는 더 올라갈 수 있어요.** 어떻게요?

이 글에서 그 방법을 다룹니다.

---

## 파라미터 vs 하이퍼파라미터

먼저 용어를 정리하고 갑시다. 헷갈리시는 분들이 많아요.

### 파라미터 (Parameter)

**모델이 학습하면서 스스로 알아내는 값**입니다.

- 선형 회귀의 가중치(weight)와 편향(bias)
- 신경망의 모든 노드 사이의 연결 강도
- SVM의 서포트 벡터들

이건 우리가 정하는 게 아니에요. **`fit()` 안에서 모델이 데이터를 보고 알아냅니다.**

### 하이퍼파라미터 (Hyperparameter)

**우리가 학습 전에 정해 줘야 하는 값**입니다.

- SVM의 `C`, `kernel`, `gamma`
- Random Forest의 `n_estimators`, `max_depth`
- KNN의 `n_neighbors`

이걸 어떻게 정하느냐에 따라 모델 성능이 크게 달라져요. 이 값들을 잘 찾는 일이 **하이퍼파라미터 튜닝**입니다.

### 비유

축구 감독을 떠올려 봐요.
- **파라미터**: 선수들의 실제 위치 (시합 중 실시간으로 결정)
- **하이퍼파라미터**: 포메이션 (시합 전에 미리 결정. 4-3-3 vs 4-4-2)

---

## 가장 단순한 방법: 손으로 바꾸기

```python
# 시도 1
model = SVC(C=1.0, kernel='rbf')
# ... 학습 ...
# 정확도: 0.91

# 시도 2
model = SVC(C=10.0, kernel='rbf')
# ... 학습 ...
# 정확도: 0.93

# 시도 3
model = SVC(C=100.0, kernel='rbf')
# ... 학습 ...
# 정확도: 0.92
```

이렇게요. 처음에는 손으로 바꿔 가며 직관을 키우는 것도 좋아요. 그런데 **하이퍼파라미터가 5개쯤 되면 조합이 폭발**합니다.

```
C: 5가지
gamma: 5가지
kernel: 3가지
→ 5 × 5 × 3 = 75가지 조합
```

손으로는 무리예요. 자동화 도구가 필요합니다.

---

## Grid Search: 모든 조합 다 시도

`GridSearchCV`는 **여러분이 지정한 모든 조합**을 다 시도해 봐요.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 시도할 값들
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear'],
}

# 그리드 서치 객체 생성
grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,                # CPU 다 쓰기
    verbose=2,                # 진행상황 출력
)

# 학습 (모든 조합을 다 돌려봄)
grid_search.fit(X_train, y_train)

# 가장 좋은 결과
print("최고 정확도:", grid_search.best_score_)
print("최고 조합  :", grid_search.best_params_)

# 가장 좋은 모델
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

```
최고 정확도: 0.945
최고 조합  : {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
```

### 옵션들

- `cv=5`: 교차검증 (Cross-Validation) 5번. 잠시 후에 설명.
- `scoring='accuracy'`: 평가 지표 ('f1', 'roc_auc' 등도 됨)
- `n_jobs=-1`: 모든 CPU 코어 사용 (빠르게)

### 단점

조합이 많아지면 매우 느려요. 위 예시는 **3 × 3 × 2 = 18 가지** 조합이고, 각각 `cv=5` 만큼 학습하니까 **90번** 학습합니다. SVM 학습이 한 번에 1분 걸리면, 90분이에요.

---

## Random Search: 무작위 조합

`RandomizedSearchCV`는 **모든 조합을 다 시도하지 않고, 무작위로 N개**만 시도해요.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# 분포(distribution)로 지정 (단순 리스트도 됨)
param_dist = {
    'C': loguniform(0.1, 100),       # 0.1~100 사이 로그 균일분포
    'gamma': loguniform(0.001, 1),
    'kernel': ['rbf', 'linear'],
}

random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=20,                # 20번만 시도
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2,
)

random_search.fit(X_train, y_train)

print("최고 정확도:", random_search.best_score_)
print("최고 조합  :", random_search.best_params_)
```

### Grid vs Random — 어느 게 더 좋아요?

직관과는 반대로, **Random Search가 보통 더 효율적**이에요. 이유는:

1. **차원의 저주**: 하이퍼파라미터가 5개 이상이면 Grid는 폭발적으로 늘어남
2. **중요한 파라미터에 집중**: Random Search는 중요한 1~2개 파라미터의 다양한 값을 시도하게 됨
3. **시간 제약 가능**: `n_iter`로 시간 예산 통제 가능

### 권장

```
탐색 시작 단계   → Random Search (n_iter=20~50)
좁힌 후 미세조정 → Grid Search (좁은 범위로)
```

---

## Cross-Validation (교차검증)

위 코드에서 `cv=5`라고 나왔죠. 이게 뭔지 짚고 가야 해요.

### 문제 상황

```python
# 단순한 방법
X_train, X_test = ...
model.fit(X_train)
score = model.score(X_test)
```

이렇게 하면 **운에 좌우돼요.** 우연히 test 데이터가 쉬운 것들로 잡히면 점수가 높게 나오고, 어려운 것들로 잡히면 낮게 나오죠.

### 해결: K-Fold Cross-Validation

데이터를 **K개로 쪼개서 K번 학습**하고, 매번 다른 부분을 test로 써요.

```
원본 데이터를 5개로 쪼갠다 (K=5)

[A][B][C][D][E]

회 1: train=BCDE, test=A → 점수1
회 2: train=ACDE, test=B → 점수2
회 3: train=ABDE, test=C → 점수3
회 4: train=ABCE, test=D → 점수4
회 5: train=ABCD, test=E → 점수5

최종 점수 = 평균(점수1, ..., 점수5)
```

이렇게 하면 **운의 영향이 줄어들고, 더 안정적인 점수**가 나옵니다.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    SVC(kernel='rbf'),
    X_train, y_train,
    cv=5,
    scoring='accuracy',
)

print("각 fold 점수:", scores)
print(f"평균 ± 표준편차: {scores.mean():.4f} ± {scores.std():.4f}")
```

```
각 fold 점수: [0.92 0.94 0.91 0.93 0.95]
평균 ± 표준편차: 0.9300 ± 0.0141
```

`cv=5`나 `cv=10`이 표준이에요. 데이터가 매우 적으면 K를 더 크게(20, 50) 쓸 수도 있어요.

### CV가 GridSearchCV의 "CV"

이제 `GridSearchCV`의 `CV`가 뭔지 아시겠죠? **모든 조합에 대해 교차검증**을 하는 거예요. 그래서 더 신뢰할 수 있는 평가 점수가 나옵니다.

---

## 실전 패턴: 처음부터 끝까지

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint


# 1. 데이터 준비
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:3000] / 255.0
y = mnist.target[:3000].astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2. 튜닝할 하이퍼파라미터 분포 정의
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],
}


# 3. RandomizedSearch
search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)


# 4. 결과 확인
print("\n[최고 결과]")
print(f"교차검증 정확도: {search.best_score_:.4f}")
print(f"최고 조합: {search.best_params_}")


# 5. 최고 모델로 test 평가
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n[Test 평가]")
print(classification_report(y_test, y_pred))
```

이 코드 한 번이면 **자동으로 모든 게 다 됩니다.** 30가지 조합을 5-fold로 시도하니까 총 150번 학습. Random Forest는 빨라서 몇 분 안에 끝나요.

---

## 모델별 주요 하이퍼파라미터

처음 튜닝하실 때 어떤 값을 건드려야 할지 막막하시죠? 모델별로 정리해 드립니다.

### Logistic Regression
```python
{
    'C': [0.001, 0.01, 0.1, 1, 10, 100],   # 정규화 강도 (작을수록 강함)
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['lbfgs', 'saga'],
}
```

### KNN
```python
{
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],   # 1=맨해튼 거리, 2=유클리드 거리
}
```

### SVM
```python
{
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly'],
}
```

### Random Forest
```python
{
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
}
```

### XGBoost
```python
{
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
```

---

## 더 나은 도구들

`RandomizedSearchCV`보다 똑똑한 도구들도 있어요. 이름만 알아두세요.

| 도구 | 특징 |
|------|------|
| **Optuna** | 베이지안 최적화 기반. RandomizedSearchCV보다 똑똑함. 정말 자주 씀. |
| **Hyperopt** | Optuna 이전 세대. 비슷한 컨셉. |
| **Ray Tune** | 분산 학습 환경에서 잘 됨. |

회사에서 진지하게 모델 만드시면 보통 **Optuna**를 씁니다. 인터페이스도 간단해요.

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

지금은 `RandomizedSearchCV`로 충분합니다. Optuna는 시간 나실 때 봐 두세요.

---

## 튜닝의 함정 — 과적합

⚠️ 이건 정말 중요해요.

`GridSearchCV`나 `RandomizedSearchCV`로 좋은 정확도를 얻었다고, 그게 **실제 성능을 보장하지는 않아요.**

왜냐하면, 우리가 **train 데이터에 대해서만 튜닝**한 거예요. 운 좋게 train 데이터에 잘 맞는 조합을 찾은 거지, **새로운 데이터에서도 잘 한다는 보장은 없어요.**

그래서 **마지막에는 반드시 test 데이터로 한 번 더 평가**해야 해요. test는 튜닝에 절대 쓰면 안 됩니다.

```python
# ❌ 잘못된 패턴
search.fit(X, y)   # 모든 데이터로 튜닝
# 결과: "와! 99% 정확도!"
# 실제로 새 데이터 들어오면 80%

# ✅ 올바른 패턴
X_train, X_test, ... = train_test_split(X, y, ...)
search.fit(X_train, y_train)   # train만 사용
# CV 점수: 0.94
final_score = search.best_estimator_.score(X_test, y_test)
# Test 점수: 0.92  ← 이게 진짜 성능
```

---

## 정리

```
1. 베이스라인 모델 학습 (기본 옵션)
   ↓
2. RandomizedSearchCV로 넓게 탐색 (n_iter=30~50)
   ↓
3. 좁힌 범위로 GridSearchCV (선택)
   ↓
4. 최고 모델로 test 데이터 평가  ← 진짜 점수
   ↓
5. 결과 만족 → 모델 저장
   불만족 → 모델/특성 다시 시도
```

**한 줄 요약: "RandomizedSearchCV로 시작하시고, test는 마지막 한 번만 보세요."**

---

## 자주 묻는 질문

> **Q. 튜닝하면 정확도가 항상 오르나요?**
>
> 보통 1~3% 정도 오릅니다. 가끔 5%까지도 올라요. 하지만 데이터가 부족하거나 모델 종류 자체가 안 맞으면, 튜닝해도 큰 변화 없어요. 그땐 **모델 자체나 특성을 바꿔야** 합니다.

> **Q. n_iter는 얼마로 해야 해요?**
>
> 시간이 허락하는 만큼. 보통 20~50번. 100번 이상 가도 큰 차이는 잘 안 나요.

> **Q. cv=5와 cv=10 중 어느 게 좋나요?**
>
> 5가 표준이에요. 데이터가 적으면 (~수백 개) 10을 쓰기도 합니다. cv가 클수록 더 정확하지만 더 느려요.

➡️ 다음: [2.8 현업 체크리스트](08-현업-체크리스트.md)
