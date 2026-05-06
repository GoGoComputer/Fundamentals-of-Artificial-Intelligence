# 5.2 과적합이라는 적

> **선행 학습:** 과적합을 이해하려면 **분산(Variance)** 개념이 도움이 돼요.  
> [👉 확률통계 입문 (5.1 평균과 분산)](../../부록/수학/05-확률통계-입문.md)  
> 단, 직관으로 먼저 이해해도 괜찮습니다.

---

## 한 학생의 비극

이 글은 한 학생의 이야기로 시작합니다.

> 어떤 학생이 시험 공부를 했어요. 100문제짜리 모의고사 한 권을 100번 풀었습니다. 결국 모든 문제의 답을 그대로 외워버렸어요.
>
> 모의고사 점수: **100점**
>
> 자신만만하게 진짜 시험을 봤습니다. 문제가 살짝 다르게 나왔어요.
>
> 진짜 시험 점수: **40점**

이게 **과적합(overfitting)** 이에요.

신경망도 똑같은 일을 합니다. 학습 데이터를 너무 잘 외워서, 새 데이터에선 망해요.

---

## 신경망은 왜 과적합되기 쉬운가요?

신경망은 **표현력이 매우 강해요.** 충분히 큰 신경망은 **거의 어떤 함수든 흉내 낼 수 있다**는 게 수학적으로 증명되어 있습니다 (Universal Approximation Theorem).

이건 양날의 검이에요.
- 좋은 패턴이 있으면 → 그 패턴을 잘 학습
- 좋은 패턴이 없으면 → 데이터의 노이즈까지 학습

### 그림으로

데이터 포인트가 다음과 같이 있다고 해 봐요. 진짜 패턴은 **부드러운 곡선**.

```
y │
  │      ●        ●
  │   ●     ●  ●
  │ ●     ●
  │
  └───────────────→ x
```

좋은 모델 (왼쪽) vs 과적합 모델 (오른쪽):

```
좋은 모델:                 과적합 모델:

y │                         y │
  │      ●        ●           │      ●        ●
  │ ─────●─────●─●            │   ●╲   ╱●  ●
  │   ●     ●                  │ ●  ╲ ╱  ╲ ╱
  │ ─●───────                  │     X    X
  └───────────                 └───────────
   부드러운 추세                   모든 점을 지나려 함
```

좋은 모델은 새 점이 와도 잘 예측해요. 과적합 모델은 새 점이 오면 엉뚱한 답을 줘요.

---

## 과적합의 신호

### 신호 1: Train과 Test 점수 차이

가장 명확한 신호예요.

```
좋은 학습:
  Train Loss: 0.05    Test Loss: 0.08    (차이 0.03)

과적합:
  Train Loss: 0.01    Test Loss: 0.50    (차이 0.49)
```

Train은 매우 낮은데 Test가 높으면 과적합.

### 신호 2: 학습 곡선

```
Loss
 │
 │\____ Train ↓↓↓ (계속 떨어짐)
 │
 │── Test (정체) 또는 ↗ (오히려 올라감)
 │
 └────────────────→ Epoch
```

이런 모양이 보이면 과적합이에요.

### 신호 3: 모델이 너무 큰가?

데이터에 비해 모델이 너무 크면 과적합 위험.

```
데이터: 1,000개
모델 파라미터: 100,000,000개   ← 과적합 거의 확실
```

대략적인 가이드: **파라미터 수 < 학습 샘플 수의 10~100배**

---

## 일부러 과적합 만들어 보기

이론만으론 안 와닿죠. 실제로 과적합을 일으켜 봅시다.

### 작은 데이터 + 큰 모델

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 진짜 패턴: y = sin(x) + noise
np.random.seed(42)
torch.manual_seed(42)

n_samples = 30   # 일부러 적게!
X = np.linspace(0, 2*np.pi, n_samples)
y = np.sin(X) + np.random.normal(0, 0.1, n_samples)

X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y, dtype=torch.float32)


# 너무 큰 모델 (과적합 유도)
class TooBigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x).squeeze()


model = TooBigModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# 너무 많이 학습 (과적합 유도)
for epoch in range(5000):
    optimizer.zero_grad()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}: Loss {loss.item():.6f}")


# 시각화
X_test = np.linspace(0, 2*np.pi, 200)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    y_pred = model(X_test_t).numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='red', label='학습 데이터', zorder=5)
plt.plot(X_test, np.sin(X_test), 'g--', label='진짜 함수 (sin)', linewidth=2)
plt.plot(X_test, y_pred, 'b-', label='모델 예측', linewidth=2)
plt.legend()
plt.title('과적합 모델')
plt.grid(alpha=0.3)
```

이걸 돌려보시면, 모델이 **모든 학습 데이터 점을 정확히 통과하지만, 그 사이에서 미친 듯이 흔들리는** 곡선을 그려요. 이게 과적합의 시각적 모습이에요.

---

## 과적합 방지의 세 가지 무기

이 장의 나머지 글에서 배울 세 가지 도구입니다.

### 1. Early Stopping (다음 글)

검증 loss가 안 좋아지면 학습을 일찍 중단.

```
Loss
 │
 │\____ Train (계속 떨어짐)
 │
 │── Test ──↘    여기서 멈춤! ←
 │      ───___
 │
 └────────────────→ Epoch
        ↑
      여기까지만 학습
```

### 2. Dropout

학습 시 일부 뉴런을 무작위로 끔.

```
   ●─●─●        ●─×─●
   │ │ │   →   │   │      (×는 꺼진 뉴런)
   ●─●─●        ●─●─×
```

매번 다른 부분이 꺼지므로 모델이 특정 뉴런에 의존하지 않게 됩니다. 마치 여러 모델의 앙상블처럼 동작.

### 3. Weight Decay (L2 정규화)

가중치가 커지지 못하게 페널티.

```
손실 = MSE + λ × Σwᵢ²
       ↑              ↑
     원래 손실      가중치 페널티
```

가중치가 작아지면 모델이 부드러워지고, 데이터의 노이즈에 덜 민감해져요.

---

## "그냥 데이터를 더 모으면 되지 않나요?"

맞아요! **데이터를 더 모으는 게 가장 강력한 과적합 방지**입니다.

```
데이터 100개 + 큰 모델 → 과적합 위험
데이터 100만 개 + 큰 모델 → 과적합 거의 없음
```

ChatGPT가 그렇게 큰 모델인데도 일반화가 잘 되는 건 **인터넷 전체로 학습**했기 때문이에요.

하지만 현실에서 데이터를 더 모으는 건 어려워요.
- 비용 (라벨링 비용 등)
- 시간
- 법적 제약 (의료, 개인정보)

그래서 우리는 **있는 데이터로 최대한 잘 학습**하는 도구가 필요한 거예요. 그게 정규화 기법들입니다.

---

## 진단 체크리스트

학습 끝난 모델에 대해:

```
[ ] Train Loss와 Test Loss 차이가 0.1 이상인가?
    → 과적합 의심
    
[ ] Train Acc가 99%+이고 Test Acc가 80% 이하인가?
    → 명확한 과적합
    
[ ] 학습 곡선에서 Test Loss가 올라가는 시점이 있나?
    → Early Stopping 필요

[ ] 모델 파라미터 수 / 데이터 수 비율이 100 이상인가?
    → 모델이 너무 큼

[ ] Dropout이나 BatchNorm 같은 정규화 층이 없나?
    → 추가 시도
    
[ ] weight_decay가 0인가?
    → 0.001~0.01 정도 시도
```

이런 신호가 보이면 다음 글들에서 배울 무기를 적용하세요.

---

## 정리

- **과적합**: 학습 데이터를 외워서 새 데이터에 못함
- **신호**: Train ≪ Test, 학습 곡선의 분기, 모델이 너무 큼
- **무기 3종**: Early Stopping, Dropout, Weight Decay
- **궁극의 해결책**: 데이터를 더 모으기

➡️ 다음: [5.3 무기 1: Early Stopping](03-Early-Stopping.md)
