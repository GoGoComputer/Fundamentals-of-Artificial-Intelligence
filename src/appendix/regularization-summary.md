# 부록: 정규화 기법 총정리

> 머신러닝/딥러닝의 정규화(과적합 방지) 기법을 한 곳에 모아 비교했습니다.
> 막힐 때마다 펴 보세요.

---

## "정규화"라는 단어의 두 가지 의미

먼저 용어 정리. 영어로 다른 두 단어가 한국어로 똑같이 "정규화"로 번역되어 헷갈려요.

| 한국어 | 영어 | 의미 | 어디서? |
|--------|------|------|--------|
| **정규화** | **Normalization / Scaling** | 데이터의 단위/범위를 맞춤 | 전처리 단계 |
| **정규화 / 규제** | **Regularization** | 모델이 너무 복잡해지지 않게 페널티 | 학습 단계 |

이 글은 **두 번째 (Regularization)** 입니다. 과적합 방지 기법들이에요.

(첫 번째 Normalization은 [3장 정규화](../03-머신러닝-회귀/이론/05-정규화.md)에서 다룸)

---

## 과적합 방지 기법 9가지 — 한눈에

| 기법 | 적용 모델 | 효과 | 비용 | 우선순위 |
|------|---------|------|------|---------|
| **데이터 더 모으기** | 모든 모델 | ⭐⭐⭐⭐⭐ | 매우 비쌈 | 1 |
| **L2 / Weight Decay** | 거의 모든 모델 | ⭐⭐⭐ | 거의 0 | 2 |
| **L1 / Lasso** | 선형 모델 위주 | ⭐⭐⭐ | 거의 0 | 3 |
| **Dropout** | 신경망 | ⭐⭐⭐⭐ | 거의 0 | 2 |
| **Batch Norm** | 신경망 | ⭐⭐⭐ | 작음 | 2 |
| **Early Stopping** | 신경망 + 일부 ML | ⭐⭐⭐⭐ | 0 | 1 |
| **데이터 증강** | 비정형 (이미지/텍스트) | ⭐⭐⭐⭐ | 작음 | 2 |
| **Label Smoothing** | 분류 신경망 | ⭐⭐ | 0 | 3 |
| **Ensemble** | 모든 모델 | ⭐⭐⭐⭐ | 비쌈 | 3 |

---

## 1. L2 정규화 (Weight Decay / Ridge)

### 무엇?

가중치의 제곱을 페널티로:

```
손실 = MSE + λ × Σwᵢ²
```

가중치가 너무 커지지 못하게 강제. 모든 가중치를 작게 (0으로는 안 함).

### 적용

```python
# sklearn
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# PyTorch
optimizer = torch.optim.AdamW(
    model.parameters(),
    weight_decay=0.01,
)
```

### 언제?

**거의 항상.** 매우 안전한 기본 정규화. 모르겠으면 추가.

### 강도

- `alpha=0.001`: 약함
- `alpha=0.01`: 표준
- `alpha=0.1`: 강함

---

## 2. L1 정규화 (Lasso)

### 무엇?

가중치의 절댓값을 페널티로:

```
손실 = MSE + λ × Σ|wᵢ|
```

**일부 가중치를 정확히 0으로** 만듦 → 자동 특성 선택.

### 적용

```python
# sklearn
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)

# PyTorch (직접 추가)
loss = mse_loss + alpha * sum(p.abs().sum() for p in model.parameters())
```

### 언제?

- 특성이 많은데 일부만 진짜 중요할 때
- 특성 선택이 필요할 때

---

## 3. ElasticNet (L1 + L2)

### 무엇?

두 정규화를 합친 것:

```
손실 = MSE + α × (l1_ratio × Σ|w| + (1-l1_ratio) × Σw²)
```

### 적용

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### 언제?

특성들이 그룹을 이뤄 상관관계가 있을 때.

---

## 4. Dropout

### 무엇?

학습 시 일부 뉴런을 무작위로 끔.

### 적용

```python
# PyTorch
self.dropout = nn.Dropout(p=0.3)

def forward(self, x):
    x = torch.relu(self.fc(x))
    x = self.dropout(x)    # 30% 확률로 끔
    return self.out(x)
```

### 언제?

거의 모든 신경망. 특히 큰 모델.

### 강도

- `p=0.1`: 약함
- `p=0.3`: 표준
- `p=0.5`: 강함 (Hinton 원논문 권장)

### 주의

`model.eval()` 모드에서 자동으로 꺼짐. 안 그러면 추론 결과가 매번 다름.

---

## 5. Batch Normalization

### 무엇?

각 층의 출력을 배치 단위로 정규화.

### 적용

```python
# PyTorch
self.bn = nn.BatchNorm1d(256)

def forward(self, x):
    x = self.fc(x)
    x = self.bn(x)         # 정규화
    x = torch.relu(x)
    return x
```

### 언제?

거의 모든 신경망. 특히 깊은 모델.

### 효과

- 학습 안정화 (1차)
- 약한 정규화 (2차)
- 더 큰 학습률 가능
- 학습 속도 향상

### 주의

- `model.eval()` 모드에서 동작 다름
- 작은 배치(<8)에서는 약함 → LayerNorm 고려
- Transformer는 LayerNorm 사용

---

## 6. Early Stopping

### 무엇?

검증 loss가 일정 epoch 동안 안 좋아지면 학습 중단.

### 적용

```python
patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(n_epochs):
    train_one_epoch()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        counter += 1
        if counter >= patience:
            break

model.load_state_dict(torch.load('best.pth'))
```

### 언제?

**모든 신경망 학습.** 사실상 표준.

---

## 7. 데이터 증강 (Data Augmentation)

### 무엇?

데이터를 인위적으로 변형해서 늘림.

### 적용 (이미지 예)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

### 언제?

- 이미지 (효과 매우 큼)
- 텍스트 (효과 보통)
- 음성 (효과 큼)
- 표 데이터 (어려움)

### 주의

평가 시엔 증강 X. 도메인 지식 필요 (좌우 반전이 의미를 바꾸는지).

---

## 8. Label Smoothing

### 무엇?

분류에서 정답을 100%가 아닌 90%로:

```
원래: [0, 0, 1, 0, 0]
스무딩: [0.025, 0.025, 0.9, 0.025, 0.025]
```

### 적용

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 언제?

분류 신경망. 매우 자만하지 않게.

---

## 9. Ensemble (앙상블)

### 무엇?

여러 모델의 예측을 평균/투표.

### 적용

```python
# 단순한 방식
preds = []
for model in models:
    preds.append(model.predict(X_test))

final_pred = np.mean(preds, axis=0)   # 또는 다수결
```

또는:
- **Bagging**: 같은 알고리즘, 다른 데이터 (Random Forest)
- **Boosting**: 순차적 보완 (XGBoost)
- **Stacking**: 모델들의 출력을 입력으로 또 다른 모델

### 언제?

- 정확도 1~2% 더 짜내야 할 때
- 캐글 대회

### 비용

학습/추론 비용이 N배.

---

## 어떤 조합이 좋아요?

### 신경망 분류 (이미지)

```
Data Augmentation + Dropout + BatchNorm + Weight Decay + Early Stopping
```

거의 표준 조합.

### 신경망 회귀 (정형)

```
Dropout + BatchNorm + Weight Decay + Early Stopping
```

### 전통 ML (sklearn)

```
- 선형: L2 (Ridge) 또는 Lasso
- 트리: max_depth, min_samples_leaf 제한
- SVM: 적절한 C 값
- 모두 공통: Cross-Validation으로 검증
```

### 작은 데이터 (< 1000개)

```
- 단순한 모델 사용 (Logistic, KNN)
- Cross-Validation
- Bootstrap (= Bagging)
- 강한 정규화
```

---

## 적용 순서 가이드

```
1. 베이스라인 학습
   ↓
2. 학습 곡선 확인
   ├─ Train ≈ Test (과소적합) → 모델 키우기
   ├─ Train ≫ Test (과적합)  → 정규화 추가
   └─ Train ≈ Test 둘 다 좋음 → ✅ 끝
   
3. 과적합인 경우, 다음 순서로 추가:
   ① Early Stopping (가장 쉬움)
   ② Dropout (PyTorch)
   ③ Weight Decay
   ④ Data Augmentation (이미지/텍스트)
   ⑤ Batch Normalization
   ⑥ 모델 줄이기
```

**한 번에 다 추가하지 마세요.** 하나씩 추가하면서 효과 확인.

---

## 자주 묻는 질문

### Q. Dropout과 BatchNorm을 같이 쓰면 어떻게 돼요?

**같이 쓸 수 있지만 순서가 중요해요.** 권장 순서:

```python
Linear → BN → ReLU → Dropout
```

또는 (덜 권장):

```python
Linear → BN → ReLU
```

BatchNorm 자체에 약한 정규화 효과가 있어서, BN만 써도 충분한 경우 많음.

### Q. Weight Decay와 L2 Regularization이 같은 거예요?

수학적으로는 같지만, **AdamW와 Adam에서 처리 방식이 살짝 달라요.** AdamW가 더 정확하게 weight decay를 적용. 그래서 AdamW 권장.

### Q. 정규화를 너무 많이 쓰면 어떻게 돼요?

**과소적합**이 됩니다. 학습이 안 돼요. Loss가 안 떨어져요.

→ 정규화 강도를 줄이거나, 일부 정규화를 빼세요.

### Q. 어떤 정규화가 가장 강력해요?

데이터에 따라 다르지만, 일반적으로:

1. **데이터 더 모으기** (압도적)
2. **데이터 증강** (비정형 데이터)
3. **Early Stopping** (모든 곳)
4. **Dropout / BatchNorm** (신경망)
5. **Weight Decay** (보조)

---

## 정리

```python
# 표준 패턴 (회사 코드 그대로)

# 모델
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)        # 정규화
        self.dropout1 = nn.Dropout(0.3)            # 정규화
        self.fc2 = nn.Linear(hidden, out_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        return self.fc2(x)


# 옵티마이저
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # 정규화
)


# Early Stopping
early_stopping = EarlyStopping(patience=10)

for epoch in range(n_epochs):
    train()
    val_loss = evaluate()
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break
```

이 한 패턴이 90%의 신경망 학습의 표준이에요.

➡️ [메인으로](../README.md)
