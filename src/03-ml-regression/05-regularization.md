# 3.5 정규화: Lasso, Ridge, ElasticNet

> **선행 학습:** 정규화의 수학적 의미를 이해하려면 **미분**과 **최적화** 개념이 도움이 돼요.  
> [👉 미분 입문](../../부록/수학/03-미분-입문.md)  
> [👉 ML에 적용하기](../../부록/수학/06-ML에-적용하기.md) — 손실 함수 최소화  
> 단, 코드부터 배우고 나중에 수학을 깊이 이해해도 괜찮습니다.

---

## 정규화가 두 가지 의미가 있어요

먼저 용어 정리부터. 영어로 둘 다 "regularization"으로 부르지만 의미가 다릅니다.

| 한국어 | 영어 | 의미 |
|--------|------|------|
| **정규화 (Normalization)** | Normalization, Scaling | 데이터의 단위를 맞춤 (StandardScaler 등) |
| **정규화 (Regularization)** | Regularization | 모델이 너무 복잡해지지 않게 페널티 |

이 글은 **두 번째**입니다. 모델이 학습 데이터에 너무 맞춰져서(과적합), 새 데이터에선 못 하게 되는 걸 막는 기법이에요.

---

## 과적합 (Overfitting)이 뭔가요?

먼저 적이 누구인지 알아야죠.

**과적합**은 모델이 학습 데이터의 **사소한 노이즈까지 다 외워 버려서**, 새 데이터엔 일반화가 안 되는 현상이에요.

### 비유

시험 공부를 한다고 해 봐요.

- 좋은 공부: 개념을 이해함 → 새 문제도 풀 수 있음
- 나쁜 공부: 답을 외움 → 본 문제만 100점, 새 문제는 0점

**과적합 = 답을 외우는 모델**입니다.

### 그래프로 보면

```
y │
  │           실제 패턴: 직선
  │      ●        ●
  │   ●     ●  ●
  │ ●     ●
  └───────────────→ x

좋은 모델 (직선):
  │      ●        ●
  │ ─●─────●──●─── ← 노이즈 무시
  │ ●     ●
  └───────────────

과적합 모델 (꼬불꼬불 곡선):
  │      ●        ●
  │ ╱╲ ●╱╲ ╱╲╱╲   ← 모든 점을 통과하려 함
  │ ●  ╲    ╲╲╱
  └───────────────
```

과적합 모델은 학습 데이터엔 100점이지만, 새 데이터엔 망해요.

### 어떻게 알아채요?

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# 차이가 크면 과적합
print(f"Train: {train_score:.4f}")
print(f"Test:  {test_score:.4f}")
print(f"Gap:   {train_score - test_score:.4f}")
```

- gap > 0.1: 과적합 의심
- gap > 0.2: 확실한 과적합

---

## 정규화의 직관

과적합을 막는 방법은 여러 가지가 있는데, 정규화는 **모델의 가중치가 너무 커지지 못하게** 페널티를 주는 방식이에요.

기본 선형 회귀의 손실 함수:
```
손실 = Σ(실제 - 예측)²
```

정규화가 추가된 손실 함수:
```
손실 = Σ(실제 - 예측)² + α × (가중치 페널티)
```

`α`(알파)는 페널티의 강도예요. 클수록 가중치가 작아지도록 강제합니다.

### 비유

학생에게 "100점 받아라" 라고만 시키면, 어떤 짓을 해서라도(컨닝, 외우기 등) 100점을 만들어요.

대신 "100점 받되, 답이 너무 길면 감점이야" 라고 하면, **단순한 풀이로 푸는 법**을 배워요.

정규화는 모델에게 "정답 맞추되, 가중치 너무 크면 손해야" 라고 말하는 거예요.

---

## 1. Ridge (L2 정규화)

가중치의 **제곱**의 합을 페널티로 줍니다.

```
손실 = MSE + α × Σwᵢ²
```

`Σwᵢ²` 가 사실 우리에게 익숙한 게 있어요. 한 점 `(w₁, w₂, ..., wₙ)` 에서 원점까지의 **거리의 제곱** 이거든요. 피타고라스 정리의 일반화입니다.

```
거리² = w₁² + w₂² + ... + wₙ² = Σwᵢ²
```

그래서 L2 정규화는 사실 **"가중치 벡터의 거리(크기)를 작게 유지하라"** 는 페널티예요.

가중치를 **작게**, 그러나 0이 되지는 않게 만들어요.

### 코드

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)
```

### 효과

가중치들이 작아져요. 모든 변수가 살아 있지만 영향력이 분산됩니다.

```python
# 일반 LinearRegression vs Ridge
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)
rg = Ridge(alpha=1.0).fit(X_train, y_train)

print("LinearRegression 가중치 절대값 합:", abs(lr.coef_).sum())
print("Ridge 가중치 절대값 합:           ", abs(rg.coef_).sum())
```

```
LinearRegression 가중치 절대값 합: 33.42
Ridge 가중치 절대값 합:            27.89
```

Ridge의 가중치가 더 작죠?

### 언제 써요?

- **모든 특성이 중요해 보일 때**
- **변수들 간 상관관계가 있을 때** (multicollinearity)
- 일반적인 정규화의 첫 시도

---

## 2. Lasso (L1 정규화)

가중치의 **절대값**의 합을 페널티로 줘요.

```
손실 = MSE + α × Σ|wᵢ|
```

이 작은 차이가 큰 결과 차이를 만들어요. **Lasso는 일부 가중치를 0으로 만들어요.** 즉, **자동으로 특성을 선택**합니다!

### 코드

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, random_state=42)
model.fit(X_train, y_train)
```

### 효과

```python
ls = Lasso(alpha=0.1).fit(X_train, y_train)

print("Lasso 가중치:")
for name, weight in zip(X_train.columns, ls.coef_):
    if abs(weight) < 1e-10:
        print(f"  {name:10s}: 0 (제거됨)")
    else:
        print(f"  {name:10s}: {weight:+.4f}")
```

```
Lasso 가중치:
  CRIM      : -0.0856
  ZN        : 0.0457
  INDUS     : 0 (제거됨)        ← !
  CHAS      : 1.8723
  NOX       : -8.0234
  RM        : 4.5612
  AGE       : 0 (제거됨)        ← !
  DIS       : -0.9876
  ...
```

Lasso는 INDUS, AGE 같은 변수가 별로 도움 안 된다고 판단해서 가중치를 0으로 만들었어요. **자동 특성 선택**의 효과예요.

### 언제 써요?

- **특성이 많은데, 진짜 중요한 게 일부일 것 같을 때**
- **모델을 더 단순하게** 만들고 싶을 때 (해석 쉬워짐)
- **특성 선택**을 하고 싶을 때

---

## 3. ElasticNet (L1 + L2 혼합)

Lasso와 Ridge를 **합친** 거예요.

```
손실 = MSE + α × (l1_ratio × Σ|wᵢ| + (1-l1_ratio) × Σwᵢ²)
```

- `l1_ratio = 1`: 순수 Lasso
- `l1_ratio = 0`: 순수 Ridge
- `l1_ratio = 0.5`: 반반

### 코드

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)
```

### 언제 써요?

- **둘 다 시도해 보고 싶을 때**
- **변수들이 그룹을 이뤄 상관관계가 있을 때**

---

## 셋의 비교 한눈에 보기

| | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|
| **페널티** | 가중치 제곱 | 가중치 절댓값 | 둘 다 |
| **가중치를 0으로?** | ❌ 작게만 | ✅ 일부 제거 | ✅ |
| **특성 선택?** | ❌ | ✅ | ✅ |
| **상관관계 있을 때** | 좋음 | 약함 | 좋음 |
| **적은 특성에서** | 좋음 | 매우 좋음 | 좋음 |
| **많은 특성에서** | 보통 | 좋음 | 매우 좋음 |

**기본 가이드:**
- 모르겠으면 **Ridge** 부터
- 특성이 많다 → **Lasso**
- 그룹 상관관계 있다 → **ElasticNet**

---

## alpha (정규화 강도) 고르기

`alpha`가 정규화의 강도예요.

- `alpha = 0`: 정규화 없음 (= LinearRegression)
- `alpha 작음`: 약한 정규화 (학습 데이터에 더 맞춤)
- `alpha 큼`: 강한 정규화 (모델이 더 단순)

### Cross-Validation으로 자동 찾기

sklearn에 `RidgeCV`, `LassoCV`, `ElasticNetCV`가 있어요. **자동으로 최적 alpha를 찾아 줘요.**

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge
rcv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
rcv.fit(X_train, y_train)
print(f"Ridge 최적 alpha: {rcv.alpha_}")

# Lasso
lcv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lcv.fit(X_train, y_train)
print(f"Lasso 최적 alpha: {lcv.alpha_}")
```

---

## 실전 비교 코드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


# 데이터 (전 글 코드 재사용)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = pd.DataFrame(data, columns=feature_names)
y = pd.Series(target)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4가지 모델 비교 (정규화 포함)
models = {
    "Linear":     LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),
    "Lasso":      Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 가중치 추출
    weights = pipe.named_steps['model'].coef_

    print(f"\n[{name}]")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  0이 된 가중치 수: {(np.abs(weights) < 1e-10).sum()}")

    results[name] = {'rmse': rmse, 'r2': r2, 'weights': weights}


# 가중치 비교 시각화
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(len(feature_names))
width = 0.2

for i, (name, r) in enumerate(results.items()):
    offset = (i - 1.5) * width
    ax.bar(x_pos + offset, r['weights'], width, label=name)

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('가중치 (정규화된)')
ax.set_title('4가지 회귀 모델의 가중치 비교')
ax.legend()
plt.tight_layout()
plt.show()
```

이 코드를 돌려보시면, **Lasso의 일부 가중치가 0**인 게 한눈에 보일 거예요.

---

## 정규화 = 과적합 방지의 한 방법일 뿐

정규화 외에도 과적합을 막는 방법들이 있어요.

| 방법 | 설명 |
|------|------|
| **정규화** | 가중치에 페널티 (이 글) |
| **데이터 증식** | 데이터를 더 모음 |
| **특성 선택** | 안 중요한 특성 빼기 |
| **Dropout** | 신경망에서 (4장에서 다룸) |
| **Early Stopping** | 학습을 일찍 멈추기 (4장) |
| **단순한 모델** | 트리 깊이 제한 등 |

상황에 따라 조합해서 씁니다.

---

## 정리

```python
# Ridge (모르겠으면 첫 시도)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# Lasso (특성 선택까지)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)

# ElasticNet (둘 다)
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# alpha 자동 찾기
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
```

**한 줄 요약:** "L2(Ridge)는 작게, L1(Lasso)는 0으로."

---

## 자주 묻는 질문

> **Q. alpha를 항상 1.0으로 하면 안 돼요?**
>
> 안 돼요. 데이터마다 적정 alpha가 달라요. 너무 크면 과소적합, 너무 작으면 과적합. 항상 CV로 찾으세요.

> **Q. 정규화 후에도 가중치를 해석할 수 있어요?**
>
> 부호와 상대적 크기는 의미가 있어요. 다만 **절대적인 수치 해석은 조심**해야 해요. 정규화로 가중치가 줄어든 것이지, 그 변수가 덜 중요해진 게 아닙니다.

> **Q. 트리 모델에도 정규화가 있어요?**
>
> 다른 형태로 있어요. `max_depth`, `min_samples_leaf` 같은 게 사실상 정규화 역할을 해요. 트리가 너무 깊어지지 못하게 막는 거죠.

➡️ 다음: [3.6 트리 기반 회귀](06-tree-regression.md)

---

## 실전 팁

정규화는 성능 향상 도구이면서 해석 안정화 도구입니다.

- Ridge는 계수를 부드럽게 줄여 과적합을 완화합니다.
- Lasso는 일부 계수를 0으로 만들어 특성 선택 효과를 줍니다.
- ElasticNet은 두 효과를 절충해 데이터 성격에 맞춘 균형을 제공합니다.



<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **3.5 정규화: Lasso, Ridge, ElasticNet** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **3.5 정규화: Lasso, Ridge, ElasticNet**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "3.5 정규화: Lasso, Ridge, ElasticNet 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **3.5 정규화: Lasso, Ridge, ElasticNet**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

3.5 정규화: Lasso, Ridge, ElasticNet는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
