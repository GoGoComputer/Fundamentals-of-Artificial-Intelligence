# 2.3 데이터 준비: 분할과 정규화

## 데이터를 그냥 모델에 넣으면 안 되나요?

물론 넣을 수는 있어요. 그런데 **결과가 전혀 신뢰가 안 됩니다.** 두 가지 큰 문제가 있어요.

### 문제 1: "공부한 문제"로만 평가하면

여러분이 시험 공부를 한다고 해 봐요. 어제 풀어 봤던 문제가 시험에 그대로 나오면, 100점 받기 정말 쉽잖아요. 그렇다고 그 점수가 여러분의 진짜 실력일까요?

**모델도 똑같습니다.** 학습한 데이터로 평가하면, 그건 그냥 "잘 외웠나"를 보는 거지 "진짜 실력"을 보는 게 아니에요.

→ 해결: **데이터를 학습용 / 평가용으로 분리**한다.

### 문제 2: 단위가 다른 데이터를 그냥 넣으면

집값 예측을 한다고 해 봅시다. 입력 데이터가 이래요.
- 평수 (예: 30평)
- 가격 (예: 500,000,000원)

이 두 값을 그대로 모델에 넣으면, 모델은 **숫자가 큰 가격에 압도적으로 휘둘려요.** "평수 같은 건 무시하자, 어차피 작아"라고 잘못된 판단을 합니다.

→ 해결: **데이터를 비슷한 범위로 정규화(스케일링)** 한다.

이 두 가지 준비가 머신러닝의 거의 항상 첫 단계예요. 하나씩 봅시다.

---

## 1. 데이터 분할: train / test split

### 가장 기본 패턴

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20%를 평가용으로
    random_state=42,    # 재현 가능하게
    shuffle=True        # 섞어서 나누기
)

print("학습용:", X_train.shape, y_train.shape)
print("평가용:", X_test.shape, y_test.shape)
```

이 한 함수로:
- 데이터를 잘 섞어서
- 80%는 X_train, y_train (학습용)
- 20%는 X_test, y_test (평가용)
- 으로 나눠 줍니다.

⚠️ 변수 이름과 순서를 외우세요. **`X_train, X_test, y_train, y_test`** 순서입니다. 처음에 자주 헷갈려요.

### 옵션 자세히 보기

#### `test_size`

평가용 비율이에요. **0.2 (20%)** 가 가장 흔하지만, 데이터가 많으면 0.1, 적으면 0.3을 쓰기도 해요.

```python
test_size=0.2     # 비율 (소수점)
test_size=1000    # 개수 (정수) - 정확히 1000개를 평가용
```

#### `random_state`

섞을 때 사용하는 시드입니다. **같은 random_state면 항상 같은 결과**가 나와요. 이게 왜 중요하냐면, **재현 가능성** 때문이에요.

```python
# 매번 다른 결과
train_test_split(X, y, test_size=0.2)

# 항상 같은 결과
train_test_split(X, y, test_size=0.2, random_state=42)
```

연구나 회사에서는 **항상 random_state를 고정**합니다. "어제 돌렸을 땐 88%였는데 오늘은 91%네?" 같은 일이 생기면 디버깅이 불가능해요.

42를 자주 쓰는 건 그냥 관례예요. (히치하이커의 안내서를 좋아하시는 분들이 많아서요. 0이든 7이든 21이든 다 됩니다.)

#### `stratify` (분류에서 중요)

분류에서 클래스 비율이 불균형하면 꼭 쓰세요.

```python
# 90%가 "정상", 10%가 "사기"인 데이터
train_test_split(X, y, test_size=0.2, stratify=y)
```

`stratify=y`를 주면, **train과 test가 같은 클래스 비율**을 유지하도록 분할됩니다. 안 그러면 운 나쁘게 test에 사기 케이스가 하나도 안 들어갈 수 있어요.

### train만으로 충분할까요? (검증 셋)

위에서는 train과 test 두 개로 나눴어요. 하지만 좀 더 진지하게 할 땐 **세 개**로 나눕니다.

| 종류 | 용도 |
|------|------|
| **train (학습용)** | 모델이 패턴을 배우는 데 사용 |
| **validation (검증용)** | 학습 중에 "잘 되고 있나" 확인하는 데 사용 |
| **test (평가용)** | 최종 평가에만 사용. 절대 학습에 안 씀. |

비유하자면:
- train = 평소 연습 문제
- validation = 모의고사
- test = 진짜 수능

`train_test_split`을 두 번 써서 만들 수 있어요.

```python
# 1단계: train + (val + test) 로 나눔
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2단계: train과 val로 한 번 더 나눔
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# 결과: train 60%, val 20%, test 20%
print(X_train.shape, X_val.shape, X_test.shape)
```

지금 단계에서는 train과 test 두 개만 써도 충분해요. 검증 셋은 나중에 [4장 딥러닝](../../04-딥러닝-분류/)에서 본격적으로 씁니다.

---

## 2. 정규화 (Scaling)

### 왜 정규화가 필요한가요?

타이타닉 데이터를 예로 들어볼게요. 입력 특성이 두 개 있다고 해 봅시다.

| 특성 | 값의 범위 |
|------|----------|
| 나이 (age) | 0 ~ 80 |
| 운임 (fare) | 0 ~ 500 |

이 두 값을 그대로 모델에 넣으면, 모델은 **운임이 더 중요하다**고 자동으로 가정해 버려요. 숫자가 더 크니까요. 하지만 그게 사실인지는 우리가 모르잖아요.

그래서 **모든 특성을 비슷한 범위로 맞춰** 줍니다. 그게 정규화예요.

### 두 가지 방법

#### 방법 1: Min-Max Scaling (0~1 범위로)

가장 직관적이에요. 최솟값을 0, 최댓값을 1로 맞춥니다.

```
정규화된 값 = (원래 값 - 최솟값) / (최댓값 - 최솟값)
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

MNIST 처럼 **이미 명확한 최대/최소가 있는 데이터**에 좋아요. 픽셀 값 0~255를 0~1로 만드는 거죠.

```python
# MNIST의 경우 그냥 255로 나눠도 됨
X = X / 255.0
```

#### 방법 2: Standard Scaling (Z-score)

이게 더 자주 쓰여요. **평균이 0, 표준편차가 1**이 되도록 변환합니다.

```
정규화된 값 = (원래 값 - 평균) / 표준편차
```

`평균` 은 데이터의 한가운데 값이고, `표준편차`는 데이터가 평균에서 얼마나 흩어져 있는지를 나타내는 값이에요. 그러니까 이 식이 하는 일은:

> "원래 값을 평균에서 얼마나 떨어져 있는지로 다시 표현해라. 단, 그 거리를 표준편차 단위로."

예를 들어 키 데이터의 평균이 170, 표준편차가 10이면:
- 180cm인 사람 → `(180 - 170) / 10 = 1` (평균보다 1σ 위)
- 160cm인 사람 → `(160 - 170) / 10 = -1` (평균보다 1σ 아래)

`170, 180, 160` 이라는 cm 값이 `0, 1, -1` 로 변하는 거예요. 모든 특성이 이렇게 같은 단위로 정규화되니, 모델이 어느 한 특성에 휘둘리지 않아요.

> **🧮 수학 보충** — 이 식은 **z-score (표준점수)** 라고 불러요. 통계에서 가장 기본적인 정규화. 표준편차의 의미가 궁금하시면 [부록 수학 5장 — 평균과 분산](../../부록/수학/05-확률통계-입문.md#51-평균과-분산--통계의-출발점)에 자세히.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**언제 어떤 걸 써야 하나요?**

- **MinMaxScaler**: 픽셀처럼 명확한 최대/최소가 있을 때
- **StandardScaler**: 그 외 거의 모든 경우 (특히 SVM, 로지스틱 회귀)

이렇게만 외워도 90% 맞아요.

### ⚠️ 정말 중요한 함정: fit은 train에만!

이 부분을 처음 배우시는 분들이 정말 많이 실수합니다.

```python
# ❌ 잘못된 코드
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)         # 전체 데이터로 fit!
X_train, X_test, ... = train_test_split(X_scaled, y, ...)
```

뭐가 잘못됐냐면, **test 데이터의 정보가 train으로 흘러갔어요.** scaler가 평균과 표준편차를 계산할 때 test 데이터의 값을 봤거든요.

이걸 **데이터 누수(data leakage)** 라고 해요. 평가 점수가 실제보다 높게 나오게 됩니다. 회사 가시면 이거 한 번 실수하면 큰일 나는 부분이에요.

**올바른 패턴:**

```python
# 1. 먼저 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

# 2. scaler를 train으로만 fit
scaler = StandardScaler()
scaler.fit(X_train)                # ← train만!

# 3. transform은 train과 test 둘 다
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 또는 fit_transform이라는 단축형 사용
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform 한 번에
X_test_scaled = scaler.transform(X_test)         # test는 transform만!
```

**핵심: `fit_transform`은 train에, `transform`만 test에.** 외워두세요.

### 트리 기반 모델은 정규화 안 해도 돼요

랜덤 포레스트, XGBoost 같은 트리 기반 모델은 **정규화가 필요 없어요.** 내부 동작이 값의 크기에 영향을 안 받거든요. 이건 알아두면 시간 절약에 좋아요.

| 모델 | 정규화 필요? |
|------|-------------|
| 선형 모델 (Linear, Logistic) | ✅ 필수 |
| SVM | ✅ 필수 |
| KNN (K-Nearest Neighbors) | ✅ 필수 |
| 신경망 (Neural Network) | ✅ 필수 |
| 결정 트리 (Decision Tree) | ❌ 안 해도 됨 |
| 랜덤 포레스트 (Random Forest) | ❌ 안 해도 됨 |
| XGBoost / LightGBM | ❌ 안 해도 됨 |

---

## 실전 패턴: 분할과 정규화 한 번에

이걸 매번 같은 패턴으로 해요. 외워 두시면 평생 갑니다.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(np.int64)

# 작게 잘라서 빠르게 실험
X = X[:5000]
y = y[:5000]

# 2. 분할 (먼저!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y    # 클래스 비율 유지
)

# 3. 정규화 (분할 후에!)
# 픽셀이라 MinMax(0~1)로 충분
X_train = X_train / 255.0
X_test = X_test / 255.0

# 또는 StandardScaler 사용
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

print("준비 완료:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
print(f"  X 범위 : {X_train.min():.2f} ~ {X_train.max():.2f}")
```

```
준비 완료:
  X_train: (4000, 784), y_train: (4000,)
  X_test : (1000, 784), y_test : (1000,)
  X 범위 : 0.00 ~ 1.00
```

이제 모델에 넣을 준비가 다 됐어요.

---

## 정리

```
원본 데이터
    ↓
[1] train_test_split   ← 먼저! 한 번만!
    ↓
train, test
    ↓
[2] scaler.fit(train)         ← train으로만 fit
[2] scaler.transform(train)
[2] scaler.transform(test)    ← test는 transform만
    ↓
모델로 ➡️
```

**한 줄 요약: "분할 먼저, 정규화 나중. 정규화는 train으로만 fit."**

---

## 자주 묻는 질문

> **Q. test_size는 0.2가 무조건인가요?**
>
> 아닙니다. 데이터 양에 따라 달라요.
> - 데이터가 100만 개 이상: test 1% 로도 1만 개라 충분
> - 데이터가 1만 개 정도: test 20% 가 표준
> - 데이터가 100개 정도: 교차검증을 쓰는 게 더 좋아요 (다음 챕터에서)

> **Q. random_state를 0이나 1로 해도 되나요?**
>
> 네, 어떤 숫자든 됩니다. 다만 **항상 같은 숫자를 쓰세요.** 코드 안에서 일관되게요.

> **Q. test 데이터로 정규화 fit을 하면 정말 그렇게 큰 문제예요?**
>
> 작은 문제로 시작해서 큰 문제로 끝납니다. 처음에는 정확도가 1~2% 더 높게 나와서 "와, 잘 됐다" 싶을 수 있는데, **실전에서 모델을 배포하면 결과가 다르게 나와서** 디버깅이 꼬여요. 회사에서는 이런 실수가 가장 비싼 실수예요.

➡️ 다음: [2.4 첫 모델 학습: SVM](04-첫-모델-SVM.md)
