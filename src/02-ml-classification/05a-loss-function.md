# 2.5a (보강) 손실 함수 깊게 — 5가지 비교

## "손실 함수가 그렇게 많아요?"

지금까지 자주 본 두 가지가 있죠.

- **MSELoss** — 회귀
- **CrossEntropyLoss** — 분류

근데 이게 다가 아니에요. 상황에 따라 다른 손실 함수가 더 좋을 때가 있어요. 이 글에서 **5가지 핵심 손실 함수**를 정리합니다.

---

## 손실 함수가 뭔가요? — 다시 한 번

> **모델의 답이 얼마나 틀렸는지를 점수로 매기는 함수.**

학습은 이 점수(loss)를 최소화하는 방향으로 가중치를 바꿔요. 그래서 **어떤 손실 함수를 쓰느냐가 모델이 무엇을 잘 맞추려고 노력하는지**를 결정합니다.

비유: 학교 시험에서 채점 기준이 다르면 학생의 공부 방향도 달라지는 것처럼.

---

## 회귀용 손실 함수 3종

### 1. MSE (Mean Squared Error) — 기본

```
MSE = (1/n) × Σ (y_true - y_pred)²
```

**특징:**
- 큰 오차에 큰 페널티 (제곱 때문에)
- 미분이 매끄러움 (학습 안정)
- **이상치에 매우 민감**

```python
loss_fn = nn.MSELoss()
```

**언제?** 모르겠으면 첫 번째 시도. 이상치가 적은 일반적인 상황.

---

### 2. MAE / L1Loss (Mean Absolute Error)

```
MAE = (1/n) × Σ |y_true - y_pred|
```

**특징:**
- 모든 오차에 같은 페널티 (선형)
- **이상치에 강함**
- 0에서 미분 불가능 (실용에선 큰 문제 X)

```python
loss_fn = nn.L1Loss()
```

**언제?** 데이터에 이상치가 많을 때. 또는 "평균보다 중앙값을 예측"하고 싶을 때.

---

### 3. Huber Loss — MSE와 MAE의 중간

```
        ┌ (1/2) × (y - ŷ)²              if |y - ŷ| ≤ δ
Huber = │
        └ δ × |y - ŷ| - (1/2) × δ²      otherwise
```

복잡해 보이지만 직관은 단순:
- **작은 오차**: MSE처럼 (제곱 → 부드러움)
- **큰 오차**: MAE처럼 (선형 → 이상치 영향 작음)

```python
loss_fn = nn.HuberLoss(delta=1.0)
# 또는 SmoothL1Loss (Huber와 사실상 같음)
loss_fn = nn.SmoothL1Loss()
```

**언제?** 이상치가 일부 있는데 일반 데이터는 잘 맞추고 싶을 때. **실무에서 정말 자주 씀.**

---

### 회귀 손실 한눈에

```
Loss
 │ 
 │   MSE (제곱)
 │   ↓
 │   /
 │  /
 │ /        Huber (꺾임)
 │/   _ _ _ _ /
 │  ___/
 │_/
 │___________ MAE (직선)
 └──────────────→ |오차|
       0
```

**표:**

| | MSE | MAE | Huber |
|---|---|---|---|
| 작은 오차 페널티 | 작음 | 보통 | 작음 |
| 큰 오차 페널티 | 매우 큼 | 보통 | 보통 |
| 이상치 영향 | 크게 받음 | 거의 안 받음 | 부분적 |
| 미분 매끄러움 | 매끄러움 | 0에서 안 됨 | 매끄러움 |
| 학습 안정성 | 좋음 | 보통 | **가장 좋음** |
| **사용 비율** | 80% | 5% | 15% |

---

## 분류용 손실 함수 2종

### 4. Cross-Entropy Loss

이미 [2.5 평가 지표](05-평가-지표.md), [4.3 첫 신경망](../../04-딥러닝-분류/이론/03-첫-신경망.md)에서 봤죠.

```
CE = - Σ y_true × log(y_pred)

(이진 분류는):
BCE = - [y log(p) + (1-y) log(1-p)]
```

직관: **"정답 클래스의 확률이 낮으면 큰 페널티."**

```python
# 다중 분류
loss_fn = nn.CrossEntropyLoss()    # softmax 자동 적용
output = model(x)                   # (B, n_classes), logit
loss = loss_fn(output, labels)      # labels는 정수

# 이진 분류
loss_fn = nn.BCEWithLogitsLoss()   # sigmoid 자동 적용
output = model(x)                   # (B, 1) 또는 (B,)
loss = loss_fn(output, labels.float())
```

**언제?** **거의 모든 분류 문제.** 표준이에요.

---

### 5. Hinge Loss — SVM의 손실

```
Hinge(y, ŷ) = max(0, 1 - y × ŷ)
```

`y`는 +1 또는 -1, `ŷ`은 모델 점수.

직관: **"정답을 마진 1 이상 자신 있게 맞추면 페널티 0. 아니면 페널티."**

이게 [2.4 SVM](04-첫-모델-SVM.md)에서 다룬 "안전 거리(마진)" 아이디어를 손실 함수로 표현한 거예요.

```python
loss_fn = nn.HingeEmbeddingLoss()
```

**언제?** SVM이 내부적으로 사용. 신경망에서는 거의 안 씀. 아는 것만으로도 충분.

---

## 어느 걸 써야 하나요? — 의사결정 트리

```
[1] 회귀 문제인가?
    │
    ├─ Y → 이상치가 많은가?
    │       │
    │       ├─ 거의 없음 → MSE
    │       ├─ 일부 있음 → Huber
    │       └─ 매우 많음 → MAE
    │
    └─ N → 분류 문제
        │
        ├─ 클래스 2개 → BCEWithLogitsLoss
        └─ 클래스 N개 → CrossEntropyLoss
```

**표 데이터 회귀의 90%:** MSE 또는 Huber
**모든 분류:** Cross-Entropy
**SVM:** Hinge (자동, 신경 안 써도 됨)

---

## 클래스 불균형 처리

분류에서 클래스가 99:1처럼 불균형하면, 모델이 "다 다수 클래스" 라고 답해도 손실이 작아요. 이걸 막는 방법.

### 1. Class Weight

손실 계산 시 소수 클래스에 더 큰 가중치.

```python
# 99 vs 1이라면 소수 클래스에 99배 가중치
weights = torch.tensor([1.0, 99.0])
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
```

### 2. Focal Loss (이진 분류에서 자주)

쉬운 샘플의 가중치를 자동으로 줄임.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()
```

객체 탐지(Object Detection)에서 표준.

---

## Label Smoothing — 분류의 작은 트릭

분류에서 **정답을 100%가 아니라 90%로 학습시키는** 기법.

원래: `정답 = [0, 0, 1, 0, 0]`
스무딩 후: `정답 = [0.025, 0.025, 0.9, 0.025, 0.025]`

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

효과: 모델이 너무 자만하지 않게 → 일반화 약간 개선.

---

## 사용자 정의 손실

비즈니스 상황에서는 표준 손실로 안 맞을 때가 있어요.

### 예: 재고 관리 (비대칭 비용)

- 너무 많이 예측: 재고 비용 1만원/개
- 너무 적게 예측: 기회 손실 5만원/개

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, over_cost=1.0, under_cost=5.0):
        super().__init__()
        self.over = over_cost
        self.under = under_cost
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(
            diff > 0,
            diff * self.over,        # 과예측
            -diff * self.under,       # 과소예측
        )
        return loss.mean()


loss_fn = AsymmetricLoss(over_cost=10000, under_cost=50000)
```

비즈니스 KPI에 맞춘 손실은 의외로 효과가 큽니다.

---

## 실수하기 쉬운 함정

### 함정 1: 마지막 층에 Sigmoid/Softmax + BCEWithLogitsLoss/CrossEntropyLoss

```python
# ❌ 잘못 — 시그모이드 두 번 적용됨!
self.fc = nn.Linear(64, 1)
self.sigmoid = nn.Sigmoid()

def forward(self, x):
    return self.sigmoid(self.fc(x))    # ← sigmoid

loss_fn = nn.BCEWithLogitsLoss()        # ← 또 sigmoid

# ✅ 올바름
self.fc = nn.Linear(64, 1)

def forward(self, x):
    return self.fc(x)                    # logit 그대로

loss_fn = nn.BCEWithLogitsLoss()         # 안에서 sigmoid
```

### 함정 2: 라벨 자료형

```python
# CrossEntropyLoss
labels = torch.tensor([0, 1, 2])           # ✅ Long
labels = torch.tensor([0.0, 1.0, 2.0])     # ❌ Float → 에러

# BCEWithLogitsLoss
labels = torch.tensor([0., 1., 0.])        # ✅ Float
labels = torch.tensor([0, 1, 0])           # ❌ Long → 에러
```

### 함정 3: shape 안 맞음

```python
output.shape   # (32, 10)
labels.shape   # (32,)        ← OK (CrossEntropyLoss)

output.shape   # (32, 1)
labels.shape   # (32,)        ← BCEWithLogitsLoss는 (32, 1) 또는 (32,)
                                squeeze로 모양 맞추기
```

---

## 정리표

| 손실 | 용도 | PyTorch | 특징 |
|------|------|---------|------|
| **MSE** | 회귀 | `nn.MSELoss()` | 표준, 이상치에 약함 |
| **MAE/L1** | 회귀 | `nn.L1Loss()` | 이상치에 강함 |
| **Huber** | 회귀 | `nn.HuberLoss()` | 둘의 장점, 가장 안정 |
| **Cross-Entropy** | 다중 분류 | `nn.CrossEntropyLoss()` | 분류의 표준 |
| **BCE** | 이진 분류 | `nn.BCEWithLogitsLoss()` | logit 입력 |
| **Hinge** | SVM | `nn.HingeEmbeddingLoss()` | 마진 기반 |

---

## 자주 묻는 질문

> **Q. CrossEntropyLoss가 NaN이 됩니다.**
>
> 흔한 원인:
> 1. log(0) — 모델 출력에 -inf가 있음 (BatchNorm 누락 등)
> 2. 학습률 너무 큼
> 3. 데이터에 NaN
>
> 해결: 학습률 ÷ 10, BatchNorm 추가, 데이터 점검.

> **Q. MSE vs Huber, 어떻게 골라요?**
>
> 데이터에 박스 플롯 그려보세요. 이상치가 보이면 Huber. 없으면 MSE. 모르겠으면 둘 다 시도해 보고 검증 점수 비교.

> **Q. Focal Loss는 어디에 쓰나요?**
>
> 클래스 불균형이 매우 심한 분류. 객체 탐지에서 표준. 일반 분류에서는 class_weight로도 충분.

➡️ 이전: [2.5 평가 지표](05-평가-지표.md)
➡️ 다음: [2.6 여러 모델 비교하기](06-여러-모델.md)
