# 2.6 여러 모델 비교하기

## "어떤 모델이 가장 좋아요?" 라는 질문

머신러닝을 시작하시면 누구나 한 번은 이 질문을 해요.

> "어떤 알고리즘이 가장 좋은 거예요? 그것만 쓰면 되는 거 아니에요?"

답은 좀 허무합니다. **데이터마다 달라요.** 어떤 데이터엔 SVM이 최고고, 어떤 데이터엔 랜덤 포레스트가 최고예요. 그래서 머신러닝 엔지니어가 하는 일 중 큰 몫이 **"여러 모델을 다 시도해 보고 가장 좋은 걸 고르기"** 입니다.

다행인 건, **sklearn의 모든 모델이 같은 인터페이스(`fit`, `predict`)를 쓰기 때문에** 비교가 정말 쉽다는 거예요.

이 글에서는 분류에서 가장 자주 쓰는 4가지 모델을 비교해 봅시다.

1. **로지스틱 회귀** — 가장 단순, 가장 빠름
2. **K-최근접 이웃 (KNN)** — 직관적
3. **SVM** — 강력하지만 느림
4. **랜덤 포레스트** — 거의 항상 잘 함

---

## 1. 로지스틱 회귀 (Logistic Regression)

이름은 "회귀"지만 사실은 **분류 알고리즘**이에요. (역사적인 이유로 이름이 그렇게 붙었어요.)

### 직관

선형 회귀가 직선을 그어 숫자를 예측하잖아요? 로지스틱 회귀는 직선을 그은 다음 **시그모이드 함수**를 통과시켜서 0~1 사이의 확률로 만들어요.

```
입력 → [선형 결합] → [시그모이드] → 0~1 확률
       wx + b           1/(1+e^-z)
```

### 특징

- ✅ **빠름** — 가장 빠른 분류기 중 하나
- ✅ **해석 가능** — 어떤 특성이 중요한지 weight를 보면 알 수 있음
- ✅ **확률 출력** — `predict_proba`가 잘 동작
- ❌ **단순한 패턴만** — 곡선이 필요한 데이터엔 약함

### 코드

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

⚠️ `max_iter`을 1000 정도로 주세요. 기본값(100)이 부족해서 경고가 뜰 수 있어요.

### 언제 쓰나요?

- 일단 **가장 먼저 시도해 보는 베이스라인**으로 좋음
- 데이터가 매우 많을 때 (빠르니까)
- 모델 결과를 사람이 해석해야 할 때

---

## 2. K-최근접 이웃 (K-Nearest Neighbors, KNN)

이건 **수학을 거의 안 쓰는** 알고리즘이에요. 정말 직관적입니다.

### 직관

새 데이터가 들어오면:
1. 학습 데이터 중 **가장 가까운 K개**를 찾는다
2. 그 K개의 라벨을 **다수결**로 결정

끝입니다. 진짜로요.

```
새 데이터 (?)
       ●
   ●       ×        K=3 → 가장 가까운 3개를 봄
       ?            (●, ●, ×) → 다수결로 ●!
   ●       ×
       ×
```

### 특징

- ✅ **이해하기 쉬움**
- ✅ **학습이 없음** (사실은 데이터를 그냥 저장만)
- ❌ **예측이 느림** (모든 학습 데이터와 거리 계산)
- ❌ **메모리 많이 씀** (학습 데이터 다 가지고 있어야 함)
- ❌ **정규화 필수** (거리 기반이라)
- ❌ **고차원에서 약함** (특성이 많으면 "가깝다"의 의미가 모호해짐)

### 코드

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`n_neighbors`(K값)는 **홀수**로 설정하세요. 짝수면 동률이 나올 수 있어요.

### K값 고르는 법

- K가 작으면 (1, 3): 노이즈에 민감, 과적합 위험
- K가 크면 (50, 100): 부드러운 결정 경계, 그러나 너무 커지면 평균값 비슷해짐
- 보통 5, 7, 9 정도부터 시도

### 언제 쓰나요?

- 데이터가 작을 때 (~10000개)
- 관계가 복잡하지만 "비슷한 것끼리는 같은 답"이 성립할 때
- 빠른 베이스라인

---

## 3. SVM (Support Vector Machine)

[2.4](04-첫-모델-SVM.md)에서 자세히 다뤘어요. 핵심만 다시.

### 직관
"두 그룹을 가장 안전한 거리로 가르는 선"

### 특징

- ✅ **고차원 데이터에서 강함**
- ✅ **kernel로 비선형 처리 가능**
- ❌ **데이터가 많으면 매우 느림** (수천 개 이상부터 부담)
- ❌ **확률 출력이 기본은 안 됨** (`probability=True` 필요)
- ❌ **정규화 필수**

### 코드

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

주요 옵션:
- `kernel`: 'rbf' (기본, 거의 항상 좋음), 'linear' (빠름), 'poly'
- `C`: 정규화 강도. 낮으면 부드럽고, 높으면 학습 데이터에 맞춰 짐 (1.0이 기본)

### 언제 쓰나요?

- 데이터가 작고 특성이 많을 때 (예: 텍스트 분류)
- 정확도가 가장 중요할 때
- 시간이 충분할 때

---

## 4. 랜덤 포레스트 (Random Forest)

지금까지의 4가지 중 **가장 자주 쓰이고, 가장 자주 좋은 결과**를 내는 모델이에요.

### 직관

여러 개의 **결정 트리(Decision Tree)** 를 학습시키고, 다수결로 답을 정해요.

결정 트리란?

```
       나이 > 30?
        /     \
      네       아니오
      /         \
  소득 > 5천?  학생?
   /  \         /  \
  ●    ×       ●    ×
```

스무고개와 비슷해요. 질문을 거치면서 답을 좁혀 나가는 거죠.

랜덤 포레스트는 **이런 트리를 100개쯤 만들어서** 다수결로 결정합니다.

### 특징

- ✅ **거의 항상 잘 함** (베이스라인으로 항상 시도)
- ✅ **정규화 필요 없음**
- ✅ **특성 중요도** 알려줌 (`feature_importances_`)
- ✅ **과적합에 강함** (여러 트리의 평균이라)
- ❌ **모델 크기가 큼**
- ❌ **결과 해석이 어려움** (트리가 100개인데 어떻게 보나요)

### 코드

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

주요 옵션:
- `n_estimators`: 트리 개수 (100이 기본, 많을수록 좋지만 느려짐)
- `max_depth`: 각 트리의 최대 깊이 (제한 안 두면 과적합 위험)
- `n_jobs=-1`: CPU 코어 다 쓰기 (학습 빨라짐)

### 특성 중요도

학습 후에 어떤 특성이 중요했는지 볼 수 있어요.

```python
importances = model.feature_importances_
print(importances[:10])

# 막대그래프
import matplotlib.pyplot as plt
plt.bar(range(len(importances)), importances)
plt.show()
```

### 언제 쓰나요?

- **거의 항상 첫 번째로 시도해 보세요.** 진짜로요.
- 표(테이블) 형태의 데이터
- 정확도와 안정성을 둘 다 원할 때

---

## 4가지 모델 한 번에 비교하기

같은 데이터로 4가지를 다 학습시키고 비교하는 코드입니다. 머신러닝의 표준 패턴이에요.

```python
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 데이터 준비
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:2000] / 255.0   # 2000개로 줄이고 정규화
y = mnist.target[:2000].astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 비교할 모델들
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)":           SVC(kernel='rbf', random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}


# 차례로 학습 + 평가
results = []
for name, model in models.items():
    print(f"\n[{name}]")
    
    # 학습 시간 측정
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # 예측 시간 측정
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start
    
    # 정확도
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  정확도:    {acc:.4f}")
    print(f"  학습 시간: {train_time:.2f}초")
    print(f"  예측 시간: {pred_time:.4f}초")
    
    results.append({
        "name": name,
        "accuracy": acc,
        "train_time": train_time,
        "pred_time": pred_time,
    })

# 정확도 순으로 정렬
results.sort(key=lambda r: r["accuracy"], reverse=True)
print("\n" + "=" * 60)
print(f"{'모델':<25} {'정확도':>10} {'학습(초)':>10} {'예측(초)':>10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<25} {r['accuracy']:>10.4f} "
          f"{r['train_time']:>10.2f} {r['pred_time']:>10.4f}")
```

```
============================================================
모델                          정확도    학습(초)   예측(초)
------------------------------------------------------------
SVM (RBF)                     0.9450      4.32     0.8521
Random Forest                 0.9325      1.85     0.0432
KNN (k=5)                     0.9275      0.05     1.2453
Logistic Regression           0.8950      0.42     0.0021
```

### 어떻게 읽나요?

- **정확도**: SVM이 1등, 랜덤 포레스트와 KNN이 비슷, 로지스틱이 가장 낮음
- **학습 속도**: KNN이 가장 빠름 (사실 학습을 안 함)
- **예측 속도**: 로지스틱이 가장 빠름. KNN은 매우 느림.
- **종합**: **랜덤 포레스트가 균형이 가장 좋음** (정확도도 좋고, 속도도 빠르고)

이런 표를 회사 미팅에서 자주 보게 됩니다. 정확도만 보지 마시고, **시간/메모리 같은 운영 비용**도 같이 보세요.

---

## 시각화로 비교

```python
import matplotlib.pyplot as plt

names = [r["name"] for r in results]
accs = [r["accuracy"] for r in results]
train_times = [r["train_time"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 정확도
axes[0].barh(names, accs, color='skyblue')
axes[0].set_xlabel('정확도')
axes[0].set_title('모델별 정확도')
axes[0].set_xlim(0.8, 1.0)
for i, v in enumerate(accs):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center')

# 학습 시간
axes[1].barh(names, train_times, color='salmon')
axes[1].set_xlabel('학습 시간 (초)')
axes[1].set_title('모델별 학습 시간')
for i, v in enumerate(train_times):
    axes[1].text(v + 0.05, i, f'{v:.2f}s', va='center')

plt.tight_layout()
plt.show()
```

---

## 그럼 어떻게 골라야 해요?

이렇게 골라 보세요. 회사 시작하시면 이 순서로 시도하세요.

```
1. 데이터 탐색
   ↓
2. Random Forest 한 번 돌려서 베이스라인 확인
   ↓
3. 그게 만족스러우면 → 이걸로 진행
   불만족이면:
   ├─ 데이터가 적다 → SVM 시도
   ├─ 빠른 응답이 필요하다 → Logistic Regression
   ├─ 데이터가 매우 많다 → XGBoost / LightGBM
   └─ 그래도 안 되면 → 딥러닝 (4장)
```

**현업에서의 황금 법칙: "단순한 것부터, 복잡한 것까지."** 처음부터 복잡한 모델을 쓰면 디버깅이 어렵고 비용도 많이 들어요.

---

## 다른 모델들 (참고)

이 4가지 외에도 자주 쓰이는 모델들이 있어요. 키워드만 알아두세요.

| 모델 | sklearn 클래스 | 언제? |
|------|--------------|------|
| **Naive Bayes** | `MultinomialNB`, `GaussianNB` | 텍스트 분류 |
| **Decision Tree** | `DecisionTreeClassifier` | 단일 트리 (Random Forest의 빌딩 블록) |
| **Gradient Boosting** | `GradientBoostingClassifier` | 정확도가 매우 중요할 때 |
| **XGBoost** | `xgboost.XGBClassifier` | 캐글 대회에서 가장 자주 우승 |
| **LightGBM** | `lightgbm.LGBMClassifier` | XGBoost보다 빠름 |
| **CatBoost** | `catboost.CatBoostClassifier` | 범주형 변수 자동 처리 |

XGBoost, LightGBM은 sklearn 표준에는 없지만 **거의 표준 도구**예요. 캐글이나 회사 실무에서 정말 자주 씁니다. 사용법은 sklearn과 거의 같아요.

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 패턴 똑같죠?
```

이 자료에서는 sklearn 모델만 다루지만, **개념을 익히신 다음에는 XGBoost로 자연스럽게 넘어가실 수 있어요.**

---

## 정리

```python
# 모든 sklearn 분류기는 같은 패턴
from sklearn.XXX import SomeClassifier

model = SomeClassifier(parameter1=value1, ...)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

| 모델 | 특징 | 언제? |
|------|------|------|
| Logistic Regression | 빠름, 해석 가능 | 베이스라인, 큰 데이터 |
| KNN | 직관적 | 작은 데이터 |
| SVM | 정확도 좋음, 느림 | 작은 데이터, 정확도 중요 |
| Random Forest | 거의 항상 좋음 | 첫 시도 |

**현업 팁: Random Forest로 시작하세요.**

➡️ 다음: [2.7 하이퍼파라미터 튜닝](07-하이퍼파라미터.md)
