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

➡️ 다음: [3.3 선형 회귀: 가장 단순하지만 강력한](03-linear-regression.md)

---

## 실전에서 보는 패턴

데이터 탐색 단계에서 아래를 먼저 확인하면 모델링 오류를 크게 줄일 수 있습니다.

1. 타깃 분포의 치우침 여부
2. 특성 간 상관 구조
3. 결측/이상치 존재 위치
4. 단위가 크게 다른 특성 존재 여부



<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **3.2 데이터를 만나다 (Boston Housing)** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **3.2 데이터를 만나다 (Boston Housing)**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "3.2 데이터를 만나다 (Boston Housing) 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **3.2 데이터를 만나다 (Boston Housing)**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

3.2 데이터를 만나다 (Boston Housing)는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
