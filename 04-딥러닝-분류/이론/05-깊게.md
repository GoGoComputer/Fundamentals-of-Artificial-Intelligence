# 4.5 더 깊게: 활성화/손실/옵티마이저

## 골라야 할 세 가지

신경망을 만들 때 매번 세 가지를 고르게 돼요.

1. **활성화 함수** (Activation): 어떤 비선형성을 줄까?
2. **손실 함수** (Loss): 어떻게 틀림을 측정할까?
3. **옵티마이저** (Optimizer): 어떻게 가중치를 업데이트할까?

이 글에서 각각의 선택지와 언제 뭘 써야 하는지 정리합니다.

---

## 1. 활성화 함수

### ReLU (가장 많이 씀)

```
ReLU(x) = max(0, x)

  y │
    │      /
    │    /
    │  /
   ─┼─────────── x
    │
```

```python
torch.relu(x)
# 또는
nn.ReLU()
```

- ✅ 계산이 빠름
- ✅ Gradient vanishing이 적음
- ❌ 음수 영역에서 그래디언트가 0 (Dying ReLU)
- **언제?** 거의 모든 곳의 기본값

### Leaky ReLU

```
Leaky ReLU(x) = max(0.01x, x)

  y │
    │      /
    │    /
   ─┼──/─────── x
    │
    │
```

음수도 살짝 살려둠. Dying ReLU 방지.

```python
nn.LeakyReLU(0.01)
```

### Sigmoid

```
Sigmoid(x) = 1 / (1 + e^-x)

  y │              ___
   1│           __/
    │         _/
 0.5│       _/
    │     _/
    │   _/
   0└─────────── x
```

- ❌ Gradient vanishing 심함 (큰 값에서 그래디언트가 거의 0)
- ❌ 출력이 0 중심이 아님
- **언제?** 이진 분류의 마지막 층 (확률 출력)

```python
torch.sigmoid(x)
```

### Tanh

Sigmoid랑 비슷한데 -1~1 범위.

```python
torch.tanh(x)
```

- 그래도 Sigmoid보다는 학습이 잘 됨
- **언제?** RNN의 내부 등 특수 상황

### GELU (현대 표준)

ReLU의 부드러운 버전. Transformer 시대의 사실상 표준.

```python
nn.GELU()
```

- ✅ ReLU보다 살짝 더 좋은 성능
- ✅ Transformer에서 표준
- **언제?** ReLU 대신 시도해 봐도 좋음

### Softmax (출력층 전용)

여러 값을 **합쳐서 1이 되는 확률**로 변환.

```
입력: [-0.3, 1.2, 0.8, 5.7, 0.1]
출력: [0.001, 0.05, 0.02, 0.93, 0.005]   ← 합 = 1
```

수식으로 보면:

```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

읽으면: **"i번째 출력의 softmax = (e의 zᵢ제곱) / (모든 j의 e의 zⱼ제곱의 합)."**

`e ≈ 2.718` 은 자연상수예요. 두 가지 효과를 줘요.
1. **항상 양수** (`e^x` 는 음수일 때도 양수). 그래서 확률처럼 0 이상.
2. **차이를 강조** — 점수가 살짝 더 큰 것이 확률은 훨씬 더 큼. 그래서 "확신 있는 답" 이 나옴.

분모가 모든 점수의 합이라 결과의 합은 항상 1. 깔끔하죠.

```python
torch.softmax(x, dim=-1)
```

- **언제?** 다중 분류의 마지막 층
- **주의**: PyTorch의 `CrossEntropyLoss`는 내부에 softmax가 들어 있어요. 모델에 직접 softmax를 추가하면 **이중으로 적용**됩니다. 주의!

> **🧮 수학 보충** — sigmoid (`1/(1+e⁻ˣ)`) 와 softmax 의 미분이 왜 깔끔한지, cross-entropy와 결합 시 그래디언트가 왜 그렇게 단순해지는지는 [부록 수학 6장](../../부록/수학/06-ML에-적용하기.md#63-sigmoid와-softmax--활성화-함수의-수학)에 자세히.

### 정리: 어디에 무엇을?

| 위치 | 추천 활성화 |
|------|-----------|
| 은닉층 (hidden layer) | **ReLU** (또는 GELU) |
| 이진 분류 출력 | Sigmoid (또는 그냥 logit) |
| 다중 분류 출력 | (X) — CrossEntropyLoss가 알아서 |
| 회귀 출력 | (X) — 활성화 없음 |

---

## 2. 손실 함수

### CrossEntropyLoss (분류용)

```python
loss_fn = nn.CrossEntropyLoss()

# 사용
output = model(x)              # (batch, num_classes)
loss = loss_fn(output, labels) # labels는 정수
```

수식으로 보면:

```
CE = -Σᵢ yᵢ × log(pᵢ)
```

읽으면: **"교차엔트로피 = 마이너스 (모든 i에 대해 정답 yᵢ × 예측 확률 pᵢ의 로그)의 합."**

`y` 가 one-hot (정답 위치만 1, 나머지 0) 이므로, 사실 식은 더 단순해져요.

```
CE = -log(p_정답)
```

정답 클래스의 예측 확률이 1에 가까우면 `log` 가 0에 가까워져 손실이 작아져요. 0에 가까우면 `log` 가 무한대로 가서 손실이 폭발. **정답을 자신있게 못 맞히면 큰 페널티**.

- ✅ 다중 분류의 표준
- ✅ 내부에 softmax 포함 (마지막 층에 활성화 X)
- 라벨은 **정수** (one-hot 아님)

> **🧮 수학 보충** — Cross-Entropy 가 왜 분류의 자연스러운 손실인지(MLE 와의 관계), 정보 이론의 엔트로피·KL divergence 와 어떻게 연결되는지는 [부록 수학 5a 정보 이론](../../부록/수학/05a-확률통계-심화.md#5a6-정보-이론-information-theory--ml의-비밀-무기)에 자세히.

### BCELoss / BCEWithLogitsLoss (이진 분류)

```python
# 모델 출력이 sigmoid 통과 후 (0~1 사이) → BCELoss
loss_fn = nn.BCELoss()

# 모델 출력이 sigmoid 통과 전 (logit) → BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()
```

수치 안정성 때문에 **BCEWithLogitsLoss를 권장**합니다.

### MSELoss (회귀)

```python
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)   # 둘 다 float
```

- 평균 제곱 오차
- 가장 자주 쓰는 회귀 손실

### L1Loss (= MAE)

```python
loss_fn = nn.L1Loss()
```

- 평균 절대 오차
- 이상치에 강함

### Huber Loss (Smooth L1)

MSE와 MAE를 섞은 것. 작은 오차는 MSE처럼, 큰 오차는 MAE처럼.

```python
loss_fn = nn.HuberLoss()
```

이상치가 있는 회귀에 좋아요.

### 정리

| 문제 | 손실 함수 |
|------|---------|
| 다중 분류 | `CrossEntropyLoss` |
| 이진 분류 | `BCEWithLogitsLoss` |
| 회귀 | `MSELoss` |
| 이상치 있는 회귀 | `HuberLoss` |

---

## 3. 옵티마이저

### SGD (Stochastic Gradient Descent)

가장 기본. 가중치를 그래디언트 방향으로 그냥 업데이트.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

- ✅ 단순, 메모리 적게 씀
- ❌ 학습이 느릴 수 있음
- **언제?** 큰 데이터셋, 잘 튜닝된 환경

### SGD with Momentum

이전 업데이트의 **관성**을 더해서 더 빨리 수렴.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

```
일반 SGD: 매 스텝마다 새로 방향 결정
Momentum: 이전 방향도 일부 유지
```

비유: 산을 굴러가는 공. 모멘텀이 있으면 작은 언덕에서 멈추지 않고 계속 굴러가요.

### Adam (가장 자주 씀)

각 파라미터마다 학습률을 자동 조정.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

- ✅ 거의 항상 잘 동작
- ✅ 학습률 튜닝이 덜 중요
- ❌ SGD보다 일반화가 약간 떨어진다는 연구 있음
- **언제?** 모르겠으면 기본 선택

### AdamW

Adam의 개선 버전. weight decay(L2 정규화)를 더 정확하게 처리.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

- ✅ Adam보다 일반화 좋음
- **언제?** Transformer, 큰 모델

### RMSprop

Adam의 전신. 옛날엔 자주 썼지만 요즘은 Adam 위주.

### 정리

| 상황 | 옵티마이저 |
|------|----------|
| 모르겠을 때 | **Adam** (lr=0.001) |
| 큰 모델 / 정규화 중요 | **AdamW** |
| 큰 데이터 + 충분한 시간 | **SGD with Momentum** |
| RNN | **RMSprop** 또는 Adam |

---

## 정규화 기법들 (과적합 방지)

손실/옵티마이저는 아니지만, 모델에 추가하는 다른 도구들이 있어요.

### Dropout

학습 시 일부 뉴런을 무작위로 끔.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)    # 50% drop
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)               # 학습 시에만 활성화
        x = self.fc2(x)
        return x
```

- 효과: 모델이 특정 뉴런에 의존하지 못하게 → 과적합 방지
- 일반적으로 0.2~0.5
- **`model.eval()`** 모드에서는 자동으로 끔

### Batch Normalization

각 층의 출력을 정규화.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)    # 256 차원
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

- 효과: 학습 안정화, 빠르게 수렴
- 보통 활성화 함수 직전에
- 작은 배치에서는 효과가 떨어질 수 있음

### Weight Decay (L2 Regularization)

가중치가 너무 커지지 못하게 페널티.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

[3장 정규화](../../03-머신러닝-회귀/이론/05-정규화.md)에서 본 Ridge랑 같은 개념이에요.

### Early Stopping

검증 점수가 안 좋아지면 학습을 일찍 멈춤. [5장](../../05-딥러닝-회귀/)에서 자세히.

---

## 실전 예: 개선된 MNIST 모델

지금까지 배운 걸 다 합쳐 봅시다.

```python
import torch
import torch.nn as nn

class ImprovedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


model = ImprovedMNIST().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
```

이 모델은 기본 MLP보다 조금 더 안정적이고 일반화가 잘 됩니다.

---

## nn.Sequential: 더 짧게 쓰기

위 코드가 길죠? 단순한 구조라면 `nn.Sequential`로 짧게 쓸 수 있어요.

```python
model = nn.Sequential(
    nn.Flatten(),
    
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
).to(device)
```

같은 동작인데 클래스 정의가 필요 없어요. 단순한 모델은 이렇게 쓰는 게 깔끔합니다.

복잡한 흐름(여러 갈래로 나뉘는 등)이 필요하면 `nn.Module` 클래스로 가야 해요.

---

## 정리

```python
# 활성화: 거의 항상 ReLU
nn.ReLU()
# 또는
torch.relu(x)

# 손실: 문제에 맞게
nn.CrossEntropyLoss()    # 다중 분류
nn.BCEWithLogitsLoss()   # 이진 분류
nn.MSELoss()             # 회귀

# 옵티마이저: 모르겠으면 Adam
torch.optim.Adam(model.parameters(), lr=0.001)
torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 과적합 방지
nn.Dropout(0.3)
nn.BatchNorm1d(256)
weight_decay=0.0001  # 옵티마이저 옵션
```

**기본 조합:**
```
ReLU + CrossEntropyLoss + Adam
```

이걸로 90%의 분류 문제를 풀 수 있어요.

➡️ 다음: [4.6 GPU 활용](06-GPU-활용.md)
