# 5.5 무기 3: Weight Decay (L2)

> **선행 학습:** Weight Decay는 **3장의 Ridge 정규화**와 같은 개념입니다.  
> [👉 정규화: Lasso, Ridge, ElasticNet](../../03-머신러닝-회귀/이론/05-정규화.md)  
> [👉 미분 입문](../../부록/수학/03-미분-입문.md) — 최적화 개념

---

## 머신러닝의 Ridge가 신경망에서는?

[3장 정규화](../../03-머신러닝-회귀/이론/05-정규화.md)에서 Ridge 회귀를 배웠죠. **가중치를 작게 만드는** 정규화 방식이었어요.

신경망에서도 같은 방식이 있어요. **Weight Decay** 라고 부릅니다. PyTorch 옵티마이저의 옵션 한 줄로 끝나요.

---

## 단 한 줄 추가

```python
# Weight Decay 없음
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Weight Decay 있음
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

`weight_decay=0.01` 이 한 부분이 전부예요. 진짜 한 줄.

---

## 어떻게 동작해요?

학습 시 손실 함수에 **가중치의 크기에 대한 페널티**를 더해요.

```
원래 손실: MSE
Weight Decay 추가: MSE + λ × Σwᵢ²
                          ↑
                       페널티
```

`λ` (람다)가 `weight_decay` 값이에요. 클수록 가중치를 더 작게 만들도록 강제합니다.

### 비유

학생이 답을 외워서 풀려고 해요. 그런데 선생님이 "답안지에 너무 많은 글자를 쓰면 감점이야" 라고 말합니다.

학생은 답을 외우면서도 **간단하게** 답해야 해요. 결과적으로 외우기보다는 핵심 패턴을 학습하게 됩니다.

Weight Decay가 신경망에 같은 효과를 줍니다. **단순한 모델로 학습하라고 강제**해요.

---

## weight_decay 값을 얼마로?

| weight_decay | 효과 |
|------------|------|
| 0 | 정규화 없음 |
| 1e-5 (0.00001) | 매우 약함 |
| 1e-4 (0.0001) | 약함 |
| 1e-3 (0.001) | 표준 |
| 1e-2 (0.01) | 강함 |
| 1e-1 (0.1) | 매우 강함 (학습 안 될 수도) |

**처음 시도: 1e-4 (0.0001)**

과적합이 심하면 키우고 (0.001 → 0.01), 학습이 안 되면 줄이세요.

---

## AdamW: 더 정확한 Weight Decay

`Adam`과 `AdamW`는 weight decay 처리 방식이 살짝 달라요.

| | Adam | AdamW |
|---|---|---|
| Weight Decay 처리 | 부정확 (L2 정규화로 변형됨) | 정확 (진짜 weight decay) |
| 일반화 성능 | 보통 | 더 좋음 |

**큰 모델이나 weight decay를 많이 쓸 땐 AdamW를 쓰세요.**

```python
# 권장
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

이게 현대 신경망의 표준 옵티마이저입니다 (Transformer 등).

---

## 가중치가 진짜 작아지나? 확인해 보기

```python
def get_weight_norm(model):
    """모든 가중치의 L2 norm 합."""
    norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            norm += p.norm(2).item() ** 2
    return norm ** 0.5


# 두 모델 비교
model_no_wd = MyModel()
optimizer_no = torch.optim.Adam(model_no_wd.parameters(), lr=0.001, weight_decay=0)

model_wd = MyModel()
optimizer_wd = torch.optim.Adam(model_wd.parameters(), lr=0.001, weight_decay=0.01)

# 둘 다 같은 데이터로 같은 step 만큼 학습 후
print(f"WD 없음: {get_weight_norm(model_no_wd):.4f}")
print(f"WD 0.01:  {get_weight_norm(model_wd):.4f}")
```

```
WD 없음: 145.32
WD 0.01:  87.46
```

가중치가 더 작아진 게 보여요.

---

## Weight Decay만 쓰지 마세요

Weight Decay 하나만으로는 충분하지 않아요. **Dropout, Early Stopping과 같이 쓰세요.**

세 가지를 조합해서 쓰는 게 표준 패턴입니다.

```python
# 표준 패턴
class RobustModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)    # ← Dropout
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        return self.fc3(x).squeeze()


model = RobustModel()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← Weight Decay
)

# 학습 시 Early Stopping 적용 ← 셋 다!
```

---

## 부분적으로 적용하기 (고급)

`weight_decay`를 모든 파라미터에 똑같이 적용하면 가끔 문제가 됩니다. 특히 **bias나 BatchNorm 파라미터**에는 weight decay를 적용 안 하는 게 좋아요.

```python
# bias와 BN을 빼고 weight decay 적용
decay = []
no_decay = []

for name, param in model.named_parameters():
    if 'bias' in name or 'bn' in name.lower() or 'norm' in name.lower():
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=0.001)
```

복잡해 보이지만, 큰 모델 학습할 때 표준이에요. 처음 배우실 땐 그냥 모두에 적용해도 큰 차이 없습니다.

---

## L1 vs L2 Weight Decay

PyTorch는 기본적으로 **L2 (제곱)** 만 지원해요. **L1**을 쓰고 싶으면 직접 추가해야 합니다.

### L2 (PyTorch 기본)

```python
# 자동
optimizer = torch.optim.AdamW(..., weight_decay=0.01)
```

페널티: `Σwᵢ²`

### L1 (직접 구현)

```python
loss_fn = nn.MSELoss()

# 학습 루프 안
loss = loss_fn(pred, y)
l1_penalty = sum(p.abs().sum() for p in model.parameters())
total_loss = loss + 0.001 * l1_penalty    # 0.001은 L1 강도
total_loss.backward()
```

**언제 어떤 걸?** 거의 항상 L2를 씁니다. L1은 특성 선택을 자동으로 하고 싶을 때 가끔.

---

## 정리

```python
# 1. 옵티마이저에 weight_decay 추가 (한 줄)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← 이거
)

# 2. 다른 정규화와 같이 쓰기
- Dropout (이전 글)
- BatchNorm
- Early Stopping
```

**한 줄 요약: "AdamW + weight_decay=0.01 부터 시작."**

---

## 자주 묻는 질문

> **Q. weight_decay와 Dropout 중 뭐가 더 강해요?**
>
> 둘이 다른 효과예요. 같이 쓰세요. weight_decay는 **모델을 작고 단순하게**, Dropout은 **앙상블 효과**.

> **Q. weight_decay 값이 너무 크면 어떻게 돼요?**
>
> Loss가 거의 안 떨어져요. 모델이 학습을 못 해요. 줄이세요.

> **Q. Adam과 AdamW 중 어느 걸 써요?**
>
> 모르겠으면 AdamW. 더 좋은 일반화. 단점은 거의 없어요.

➡️ 다음: [5.6 세 무기 다 같이 쓰기](06-종합.md)
