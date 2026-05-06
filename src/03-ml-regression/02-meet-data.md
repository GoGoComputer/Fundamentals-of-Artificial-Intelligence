# 3.2 데이터를 만나다 (Boston Housing)

## 회귀계의 MNIST: Boston Housing

분류에는 MNIST가 있었죠. 회귀에는 **Boston Housing 데이터셋**이 있습니다.

미국 보스턴 지역의 집값과 주변 환경 정보를 모은 데이터예요. 1970년대에 처음 공개됐고, 회귀를 가르치는 책들의 단골 예제입니다.

> ⚠️ **주의**: Boston 데이터셋은 사회적 편향(인종 변수 등)이 있어서, sklearn 1.2 이후로 기본 제공에서 빠졌어요. 교육 목적으로만 쓰시고, 실무 학습용으로는 California Housing 등 대안을 쓰시는 게 좋습니다.

---

## Boston Housing이 뭔가요?

- **샘플 수**: 506개 (보스턴 지역 506개 동네)
- **특성 수**: 13개 (각 동네의 특징)
- **타겟**: 그 동네 집값 중앙값 (단위: $1,000)

### 13개의 특성

| 약자 | 의미 |
|------|------|
| CRIM | 동네의 범죄율 |
| ZN | 25,000 sqft 이상 주거지 비율 |
| INDUS | 비상업 토지 비율 |
| CHAS | 강 옆이면 1, 아니면 0 |
| NOX | 산화질소 농도 (오염) |
| RM | 주택당 평균 방 수 |
| AGE | 1940년 이전 지어진 집 비율 |
| DIS | 보스턴 도심까지 가중 거리 |
| RAD | 고속도로 접근성 지수 |
| TAX | 재산세율 |
| PTRATIO | 학생-교사 비율 |
| B | 흑인 인구 비율 (편향 변수, 윤리적 문제) |
| LSTAT | 하층민 인구 비율 (사회적 편향) |

이 중 RM(방 수), LSTAT(저소득층 비율)이 집값과 가장 강한 상관관계가 있어요.

---

## 데이터 불러오기

sklearn 1.2 이후로는 기본 제공이 안 되니, 우리는 직접 불러와야 해요.

### 방법 1: 직접 URL에서

```python
import pandas as pd
import numpy as np

# 카네기 멜런 대학에서 호스팅
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# 데이터 정리 (좀 복잡한 형태로 저장돼 있음)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

X = pd.DataFrame(data, columns=feature_names)
y = pd.Series(target, name='MEDV')

print(X.shape, y.shape)    # (506, 13) (506,)
```

### 방법 2: California Housing (요즘 권장)

```python
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing(as_frame=True)
X = dataset.data
y = dataset.target

print(X.shape)    # (20640, 8)
```

이 자료에서는 일관성을 위해 Boston을 쓰지만, 본인의 학습용으로는 California Housing도 좋습니다.

---

## 데이터 둘러보기

```python
print("[X 첫 5행]")
print(X.head())

print("\n[기술 통계]")
print(X.describe())

print("\n[y 분포]")
print(y.describe())
```

```
[기술 통계]
            CRIM         ZN     INDUS  ...
count  506.000000  506.00000  506.000000
mean     3.613524   11.36364   11.136779
std      8.601545   23.32253    6.860353
min      0.006320    0.00000    0.460000
max     88.976200  100.00000   27.740000
```

### 핵심 관찰

- **단위가 천차만별** — CRIM은 88까지, ZN은 100까지, NOX는 0~1
- **이상치가 있어 보임** — CRIM의 max가 평균보다 25배 큼
- **타겟 y의 범위** — 5(저소득) ~ 50(부유) 사이

이런 데이터엔 **정규화가 필수**예요. 안 하면 SVM, KNN, 선형 모델이 큰 값 변수에만 영향받아요.

---

## 시각화

데이터를 그림으로 보면 직관이 생겨요.

### 1. 타겟 분포

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
plt.hist(y, bins=30, color='steelblue', edgecolor='black')
plt.xlabel('집값 (중앙값, $1000)')
plt.ylabel('동네 수')
plt.title('Boston 집값 분포')
plt.show()
```

50에서 봉우리가 있어요. 데이터의 한계(censored)일 가능성이 큽니다. (50보다 비싼 집은 다 50으로 기록)

### 2. 상관관계 히트맵

```python
import seaborn as sns

# y를 X에 합쳐서 상관관계 보기
data_full = X.copy()
data_full['MEDV'] = y

corr = data_full.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
plt.title('Boston 상관관계 행렬')
plt.tight_layout()
plt.show()
```

마지막 행/열(MEDV)을 보면, 어떤 특성이 집값과 강하게 연관됐는지 보여요.

- RM (방 수): +0.70 → 방이 많으면 집값 ↑
- LSTAT (저소득층): -0.74 → 저소득층 많으면 집값 ↓

### 3. 산점도

가장 강한 두 특성으로 산점도를 그려 봅시다.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X['RM'], y, alpha=0.5)
axes[0].set_xlabel('방 개수')
axes[0].set_ylabel('집값')
axes[0].set_title('방 개수 vs 집값 (양의 상관)')

axes[1].scatter(X['LSTAT'], y, alpha=0.5, color='salmon')
axes[1].set_xlabel('저소득층 비율 (%)')
axes[1].set_ylabel('집값')
axes[1].set_title('저소득층 비율 vs 집값 (음의 상관)')

plt.tight_layout()
plt.show()
```

오른쪽 그래프는 명확한 음의 상관관계가 보여요. 회귀 모델이 해야 할 일은 **이 패턴을 식으로 잡아내는 것**이에요.

---

## 데이터 준비

이제 모델 학습 전 표준 준비를 해 봅시다.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"학습: {X_train.shape}, 평가: {X_test.shape}")

# 2. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 학습 데이터의 평균이 0, 표준편차가 1이 됐는지 확인
print(f"평균: {X_train_scaled.mean():.4f}")    # ~ 0
print(f"표편: {X_train_scaled.std():.4f}")    # ~ 1
```

회귀에서는 보통 **`stratify` 옵션을 안 줘요.** stratify는 분류용이거든요. (y가 카테고리가 아니니까)

---

## 정리

- Boston Housing은 회귀의 표준 데이터셋
- 506개 동네, 13개 특성, 타겟은 집값 중앙값
- 특성들의 단위가 다르므로 **정규화 필수**
- RM(방 수), LSTAT(저소득층 비율)이 가장 강한 예측 변수
- 데이터 탐색 → 분할 → 정규화 → 모델로

다음은 가장 단순하지만 가장 강력한 회귀 모델, 선형 회귀입니다.

➡️ 다음: [3.3 선형 회귀: 가장 단순하지만 강력한](03-선형회귀.md)
