# 3.7 현업 체크리스트

> 회귀 특유의 함정들이 있어요. 분류와 같은 것들은 [2장 8절](../../02-머신러닝-분류/이론/08-현업-체크리스트.md)을 참고하시고, 이 글은 **회귀에서만 신경 써야 할 것들** 위주로 다룹니다.

---

## 1. 타겟 변수의 분포 확인

회귀에서 가장 먼저 확인해야 할 것: **y의 분포**

### 좋은 분포

```python
import matplotlib.pyplot as plt
plt.hist(y, bins=30)
```

- **정규 분포에 가깝다** → 그대로 진행
- **약간 비대칭이지만 봉우리 하나** → 보통 그대로 OK

### 문제 있는 분포

#### 1. 매우 비대칭 (skewed)

```
    │█
    │██
    │███
    │████
    │██████
    │█████████___
    └───────────────
    낮은값         높은값
```

이런 데이터에 일반 회귀를 쓰면 큰 값에서 잘 못 맞춰요.

**해결: log 변환**

```python
import numpy as np

# 학습 시
y_log = np.log1p(y_train)    # log(1+y) - 0이 있어도 안전
model.fit(X_train, y_log)

# 예측 시
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)    # 다시 원래 단위로
```

집값, 매출, 인구 같은 데이터에 자주 적용해요.

#### 2. 봉우리 두 개 (bimodal)

```
   │██        ██
   │███      ███
   │████    ████
   │█████   █████
   └───────────────
```

데이터에 두 그룹이 섞여 있다는 뜻이에요. 그룹별로 모델을 따로 만들거나, 그룹 변수를 특성으로 추가하세요.

#### 3. 절단된 분포 (censored)

Boston 데이터의 50처럼, 실제는 더 큰데 50으로 잘려 있는 경우. 모델이 이걸 모르고 50이 진짜인 줄 알아요. 50인 샘플을 빼거나, 별도 처리가 필요해요.

---

## 2. 이상치 (Outlier) 처리

### 발견하기

```python
import seaborn as sns
sns.boxplot(y=y)
```

또는 IQR로:

```python
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = (y < lower) | (y > upper)
print(f"이상치: {outliers.sum()}개 ({outliers.mean():.1%})")
```

### 처리 방법

| 방법 | 언제? |
|------|------|
| **삭제** | 명백한 입력 오류 |
| **변환 (log 등)** | 자연스러운 비대칭 |
| **클리핑 (값 제한)** | 모델 학습용 |
| **그냥 두기** | 진짜 데이터의 일부 |

⚠️ **함부로 삭제하지 마세요.** 진짜 중요한 케이스일 수 있어요. 부동산 모델에서 "초고가 매물"을 다 빼버리면 의미가 없잖아요.

---

## 3. 다중공선성 (Multicollinearity)

여러 변수가 비슷한 정보를 가지고 있는 경우예요. 선형 모델에서 큰 문제가 됩니다.

### 발견하기

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = X_train.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.tight_layout()
plt.show()

# 상관계수 0.8 이상인 쌍 출력
import numpy as np
high_corr = np.where(np.abs(corr) > 0.8)
for i, j in zip(*high_corr):
    if i < j:
        print(f"{corr.columns[i]} <-> {corr.columns[j]}: {corr.iloc[i,j]:.3f}")
```

### 더 정확한 측정: VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i)
              for i in range(len(X_train.columns))]
print(vif)
```

VIF가 10 이상이면 다중공선성 의심.

### 해결

1. **하나만 남기고 삭제** (해석 쉬움)
2. **PCA로 차원 축소**
3. **Ridge 사용** (다중공선성에 강함)

---

## 4. 비선형성 처리

### 진단

잔차 플롯에 패턴이 보이면 비선형성이 있다는 신호예요.

```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
```

### 해결

| 방법 | 설명 |
|------|------|
| **다항 특성** | x² 같은 새 특성 추가 |
| **트리 기반 모델** | 비선형 자연스럽게 |
| **신경망** | 복잡한 비선형성 |
| **변수 변환** | log, sqrt 등 |

```python
# 다항 특성 만들기
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 이제 선형 모델로
model = Ridge()
model.fit(X_train_poly, y_train)
```

---

## 5. 시계열 데이터 (특별 주의)

데이터가 **시간 순서**가 있다면, 보통의 train/test 분할은 **잘못**입니다.

### 잘못된 방법
```python
# ❌ 무작위 분할 (시계열에는 NO)
train_test_split(X, y, shuffle=True)
```

이렇게 하면 미래 데이터로 학습해서 과거를 예측하게 돼요. 실전에선 불가능한 일이죠.

### 올바른 방법

```python
# ✅ 시간 순으로 분할
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
```

또는 sklearn의 `TimeSeriesSplit`:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # 모델 학습 + 평가
```

### 시계열 특성 만들기

- `lag` 특성 (n시간 전 값)
- 이동 평균 (rolling mean)
- 시간 자체의 특성 (요일, 월, 시간대)

---

## 6. 예측 결과 검증

### 음수가 나오면 안 되는 경우

집값, 매출 같은 건 음수가 나올 수 없죠. 모델이 음수를 예측하면 후처리:

```python
y_pred = np.maximum(0, y_pred)    # 음수면 0으로
```

또는 **log 변환**해서 학습하면 자연스럽게 양수만 나와요.

### 정수여야 하는 경우

판매량, 인원 수 같은 경우. 반올림:

```python
y_pred = np.round(y_pred).astype(int)
```

### 범위가 정해진 경우

확률처럼 0~1이어야 하면 클리핑:

```python
y_pred = np.clip(y_pred, 0, 1)
```

---

## 7. 신뢰 구간 (Confidence Interval)

회귀에서는 **단일 예측값보다 "예측 ± 오차" 가 더 유용**한 경우가 많아요.

### 간단한 방법: 잔차의 표준편차

```python
residuals = y_train - model.predict(X_train)
std = np.std(residuals)

y_pred = model.predict(X_new)
lower = y_pred - 1.96 * std    # 95% 신뢰구간
upper = y_pred + 1.96 * std

print(f"예측: {y_pred[0]:.2f}, 95% 구간: [{lower[0]:.2f}, {upper[0]:.2f}]")
```

### 더 정확한 방법: 분위수 회귀 (Quantile Regression)

```python
from sklearn.ensemble import GradientBoostingRegressor

# 5분위수 모델
lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.05, ...)
upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.95, ...)
median_model = GradientBoostingRegressor(loss='quantile', alpha=0.5, ...)

# 각각 학습 후
lower = lower_model.predict(X_test)
upper = upper_model.predict(X_test)
median = median_model.predict(X_test)
```

---

## 8. 비즈니스 손실 함수 만들기

기본 RMSE/MAE는 **위쪽 오차와 아래쪽 오차를 똑같이** 평가합니다. 하지만 비즈니스에서는 다를 수 있어요.

### 예: 재고 관리

- 너무 많이 예측 → 재고 비용 (1만원/개)
- 너무 적게 예측 → 기회 손실 (5만원/개)

비대칭이죠. 이럴 땐 **커스텀 손실 함수**를 정의해서 그걸로 평가합니다.

```python
def asymmetric_loss(y_true, y_pred):
    over = y_pred > y_true
    cost = np.where(
        over,
        (y_pred - y_true) * 10000,      # 과예측 비용
        (y_true - y_pred) * 50000,      # 과소예측 비용
    )
    return cost.mean()

# 모델 평가
loss = asymmetric_loss(y_test, model.predict(X_test))
print(f"기대 비용: {loss:,.0f}원")
```

이런 비즈니스 지표가 결국 **임원 미팅에서 쓰이는 진짜 점수**예요.

---

## 9. 모델 모니터링 (시계열 모델 특히)

### Concept Drift

데이터 패턴이 시간에 따라 바뀌어요.

예시:
- 코로나 전후 매출 패턴
- 최신 트렌드 변화
- 경쟁사 신규 진출

### 데이터 Drift

입력 데이터의 분포가 바뀌어요.

```python
# 새 데이터의 통계와 학습 데이터 비교
print("학습 평균:", X_train.mean())
print("운영 평균:", X_recent.mean())
print("학습 표준편차:", X_train.std())
print("운영 표준편차:", X_recent.std())

# 큰 차이가 있으면 alarm
```

### 운영 RMSE 추적

```python
# 매일/매주 실제 결과와 예측을 비교
daily_rmse = []
for date in date_range:
    actual = get_actual(date)
    predicted = get_predictions(date)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    daily_rmse.append(rmse)

# RMSE가 점점 올라가면 모델 재학습 신호
```

---

## 10. 회귀 특유 체크리스트

```
[ ] y의 분포 확인 (정규성, 비대칭, 봉우리)
[ ] log/sqrt 변환 필요한지 검토
[ ] 이상치 발견 + 처리 방법 결정
[ ] 다중공선성 확인 (상관관계, VIF)
[ ] 잔차 플롯으로 비선형성 확인
[ ] 시계열이면 시간 순 분할
[ ] 예측 결과 후처리 (음수, 범위)
[ ] 신뢰 구간 / 분위수 제공
[ ] 비즈니스 손실 함수로 평가
[ ] 모델 모니터링 계획
```

---

## 3장을 마치며

회귀 챕터, 끝났습니다.

핵심을 다시 정리하면:

- ✅ 회귀는 숫자를 예측, 분류는 카테고리를 예측 (코드는 거의 같음)
- ✅ 선형 회귀는 단순하지만 해석 가능 + 베이스라인으로 좋음
- ✅ MSE, RMSE, MAE, R² — 상황에 맞게 사용
- ✅ Lasso/Ridge/ElasticNet — 과적합 방지 + 특성 선택
- ✅ Random Forest와 XGBoost — 거의 항상 정확
- ✅ 회귀만의 함정들 (분포, 이상치, 시계열, 후처리)

다음은 **딥러닝**입니다. 지금까지 본 머신러닝의 한계를 넘는 방법이에요.

➡️ **[4장. 딥러닝 — 분류](../../04-딥러닝-분류/)**
