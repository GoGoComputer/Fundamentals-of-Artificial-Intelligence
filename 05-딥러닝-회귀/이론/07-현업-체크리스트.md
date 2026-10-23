# 5.7 현업 체크리스트

> 4장 분류 체크리스트([4.7](../../04-딥러닝-분류/이론/07-현업-체크리스트.md))의 거의 모든 게 적용됩니다.
> 이 글은 **회귀 특유의 추가 항목**입니다.

---

## 1. y 정규화 잊지 마세요

회귀에서 가장 자주 만나는 함정 1번이에요.

```python
# ❌ y 그대로 (집값이 0~50 범위)
loss = MSELoss(pred, y)   # loss가 너무 큼 → 학습 불안정

# ✅ y 정규화
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

# ... 학습 ...

# 예측 후 되돌리기
pred_original = scaler_y.inverse_transform(pred.reshape(-1, 1))
```

---

## 2. 출력 활성화 함수 X

분류에선 마지막에 softmax/sigmoid를 (자동으로) 적용했지만, **회귀는 활성화 없음**이에요.

```python
# ❌
def forward(self, x):
    ...
    return torch.sigmoid(self.fc_last(x))   # 0~1로 제한됨!

# ✅
def forward(self, x):
    ...
    return self.fc_last(x).squeeze()
```

타겟이 0~1 범위로 정해져 있다면 sigmoid가 의미 있을 수 있지만, 일반적인 회귀에선 활성화 안 씁니다.

---

## 3. 손실 함수 선택

| 손실 | 언제? |
|------|------|
| `MSELoss` | 표준. 큰 오차에 큰 페널티. |
| `L1Loss` | 이상치에 강함. MAE랑 같음. |
| `HuberLoss` | MSE + L1 혼합. 이상치 있으면 좋음. |
| `MSE on log(y)` | y가 매우 비대칭일 때 |

**기본: MSELoss.** 이상치 있으면 HuberLoss로 변경.

---

## 4. 분포 변환 (필요시)

3장에서도 배웠지만, 신경망 회귀에서도 마찬가지예요.

### log 변환

타겟이 매우 비대칭(예: 집값, 매출)이면 log 변환 후 학습:

```python
y_log = np.log1p(y_train)    # log(1+y)
# 학습
# 예측 후
y_pred = np.expm1(y_pred_log)    # 원래 단위로
```

### 박스-콕스 변환

더 일반적인 변환:

```python
from scipy.stats import boxcox
y_transformed, lambda_ = boxcox(y_train + 1)
```

---

## 5. 모델 출력 후처리

### 음수가 안 되는 경우 (집값, 매출, ...)

```python
y_pred = np.maximum(0, y_pred)
```

### 정수여야 하는 경우 (인원, 판매량)

```python
y_pred = np.round(y_pred).astype(int)
```

### 범위 제한

```python
y_pred = np.clip(y_pred, 0, 100)    # 0~100으로 제한
```

---

## 6. 회귀의 평가 지표

분류는 정확도, 회귀는 다양해요. **여러 개 같이 보세요.**

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def evaluate_regression(y_true, y_pred):
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }
    return metrics
```

특히 RMSE와 MAE를 같이 보세요. **차이가 크면 이상치가 있는 것.**

---

## 7. 잔차 분석

학습 끝나면 **잔차 플롯**을 꼭 그리세요. 모델 진단의 핵심.

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 vs 예측값')

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=30)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('잔차')
plt.title('잔차 분포')

plt.subplot(1, 3, 3)
from scipy import stats
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot (정규성 확인)')

plt.tight_layout()
plt.show()
```

진단:
- 1번 그래프: 패턴이 보이면 모델이 못 잡은 패턴 있음
- 2번 그래프: 0 중심의 정규분포 → 좋음
- 3번 그래프: 직선에 가까우면 정규성 OK

---

## 8. 신뢰 구간 제공

비즈니스에서는 단일 예측보다 **"예측 ± 오차"** 가 더 가치 있어요.

### 간단한 방법: 잔차 표준편차

```python
val_residuals = y_val - model.predict(X_val)
std = np.std(val_residuals)

y_pred = model.predict(X_new)
lower_95 = y_pred - 1.96 * std
upper_95 = y_pred + 1.96 * std
```

### 더 정교: Bayesian Neural Network 또는 MC Dropout

```python
# MC Dropout: 평가 시에도 dropout 켜두고 여러 번 예측
def predict_with_uncertainty(model, X, n_samples=100):
    model.train()    # ← Dropout 켜기 (평가 모드 아님)
    
    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            preds.append(model(X).cpu().numpy())
    
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    
    return mean, std
```

매번 다른 부분이 꺼지므로 다른 예측이 나와요. 그 분포가 모델의 불확실성을 나타냅니다.

---

## 9. 시계열 회귀

시계열은 일반 회귀와 다른 주의가 필요해요. (3장 7절에서 다뤘죠)

```python
# 시간 순으로 분할
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Lag 특성 만들기
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)
df['rolling_mean_7'] = df['value'].rolling(7).mean()

# 시간 자체 특성
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
```

**시계열은 신경망 중에 LSTM이나 Transformer가 더 잘 맞아요.** MLP보다요.

---

## 10. 비즈니스 손실 함수

기본 MSE는 양/음 오차를 동등하게 보지만, 비즈니스에선 다를 수 있어요.

### 예: 재고 관리

- 너무 많이 예측: 재고 비용 1만원/개
- 너무 적게 예측: 기회 손실 5만원/개

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, over_cost=1.0, under_cost=5.0):
        super().__init__()
        self.over_cost = over_cost
        self.under_cost = under_cost
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(
            diff > 0,
            diff * self.over_cost,        # 과예측
            -diff * self.under_cost,       # 과소예측
        )
        return loss.mean()


loss_fn = AsymmetricLoss(over_cost=10000, under_cost=50000)
```

---

## 회귀 특유 체크리스트

```
[ ] y 정규화 했는가
[ ] 출력에 활성화 함수 안 줬는가
[ ] 손실 함수 선택 (MSE / Huber / L1)
[ ] 타겟 분포 확인 (log 변환 필요?)
[ ] 출력 후처리 (음수 제거, 정수화 등)
[ ] RMSE + MAE 둘 다 봤는가
[ ] 잔차 플롯 그렸는가
[ ] 신뢰 구간 제공 가능?
[ ] 시계열인가? (그렇다면 시간 순 분할)
[ ] 비즈니스 손실로 평가했는가
```

---

## 5장을 마치며

축하합니다! 5장까지 다 끝내셨습니다.

지금까지 배운 것:
- ✅ 분류 신경망 (4장)
- ✅ 회귀 신경망
- ✅ 과적합과 싸우는 3가지 무기
  - Early Stopping
  - Dropout
  - Weight Decay (L2)
- ✅ 회귀 특유의 함정과 처리

이제 마지막 챕터예요. **모델을 어떻게 진짜 서비스로 만드는가**에 대한 이야기.

➡️ **[6장. 프로덕션으로 가는 길](../../06-프로덕션으로-가는-길/)**
