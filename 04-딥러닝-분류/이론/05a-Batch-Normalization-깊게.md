# 4.5a (보강) Batch Normalization 깊게

## "BatchNorm 한 줄 추가했더니 학습이 빨라졌어요"

이전 글([4.5 더 깊게](05-깊게.md))에서 `nn.BatchNorm1d(...)` 한 줄로 BatchNorm을 썼었죠. 이 글에서는 **그게 왜 마법처럼 동작하는지** 좀 더 깊이 들여다봅니다.

수식이 살짝 나오지만, 직관 위주로 가니 따라오실 수 있어요.

---

## 문제: 깊은 신경망의 골치 — Internal Covariate Shift

### 비유로 시작합시다

여러분이 컨베이어 벨트 공장에서 일한다고 해 봐요. 앞 공정에서 부품이 일정한 크기·일정한 모양으로 들어와야 작업이 됩니다. 그런데 어느 날 갑자기 부품이 **더 크게**, 또 다른 날엔 **더 작게**, 또 어떤 날엔 **삐뚤빼뚤하게** 들어오기 시작합니다.

여러분 작업이 잘 될까요? 매일 새로 적응해야 해서 죽을 맛이겠죠.

### 신경망에서 일어나는 같은 일

학습이 진행될수록 **앞 층의 가중치가 계속 변해요.** 그러면 앞 층의 출력도 계속 변하죠. 즉, **뒷 층 입장에서는 입력 분포가 매번 달라져요.**

```
초기 학습:
    Layer 1 출력 분포: 평균 0, 표준편차 1.0    → Layer 2가 적응

100 step 후:
    Layer 1 출력 분포: 평균 5, 표준편차 3.0    → Layer 2가 다시 적응 시작 😵

200 step 후:
    Layer 1 출력 분포: 평균 -2, 표준편차 0.3   → Layer 2가 또 다시 적응 😵😵
```

이 현상을 **Internal Covariate Shift (내부 공변량 이동)** 라고 부릅니다. 깊을수록 심해져요.

### 왜 문제예요?

각 층이 **앞 층의 변화를 따라잡느라 학습이 매우 느려져요.** 그래서:
- 학습률을 작게 해야 함 (변화가 너무 크면 망함)
- 초기 가중치 설정이 매우 민감
- 깊을수록 학습이 어려움 (Vanishing Gradient도 같이 옴)

---

## 해결: Batch Normalization

2015년에 구글이 발표한 아이디어. 정말 단순한데 효과가 어마어마했어요.

> **각 층의 출력을 정규화해서 다음 층에 전달하자.**

매번 평균 0, 표준편차 1로 맞춰 주면, 뒷 층 입장에서는 입력 분포가 항상 일관돼요. 적응할 필요가 없죠.

### 수식 (살짝만)

배치 단위로 정규화합니다.

```
1. 배치의 평균 계산:
   μ_B = (1/m) × Σ xᵢ

2. 배치의 분산 계산:
   σ_B² = (1/m) × Σ (xᵢ - μ_B)²

3. 정규화:
   x̂ᵢ = (xᵢ - μ_B) / √(σ_B² + ε)

4. 스케일과 이동 (학습 가능한 파라미터 γ, β):
   yᵢ = γ × x̂ᵢ + β
```

수식이 무서워 보여도, 그냥 **"평균 빼고 표준편차로 나누고, 다시 살짝 조정"** 입니다. [3장 정규화](../../03-머신러닝-회귀/이론/05-정규화.md)에서 본 StandardScaler랑 같은 일이에요.

차이는: **층 사이에서, 매 배치마다 자동으로 한다는 것.**

---

## 왜 γ와 β가 필요해요?

수식 4번이 좀 이상해 보이죠. "정규화했는데 다시 곱하고 더한다고?"

이유: **항상 평균 0, 표준편차 1만 강제하면 모델의 표현력이 줄어들어요.** 어떤 층은 다른 분포가 더 좋을 수 있거든요.

그래서:
- `γ` (감마): 다시 스케일 조정 (학습됨)
- `β` (베타): 다시 위치 조정 (학습됨)

이렇게 모델이 **"필요하면 정규화를 무시할 수도 있게"** 자유를 줘요. 똑똑한 설계예요.

---

## 효과 — 진짜로 일어나는 일

같은 모델, 같은 데이터로 BN 있고 없고 비교하면:

| | BN 없음 | BN 있음 |
|---|---|---|
| 학습률 | 0.001 (작게 해야 함) | 0.01 (10배 가능) |
| 수렴 속도 | 100 epoch | 30 epoch |
| 초기화 민감도 | 매우 민감 | 덜 민감 |
| 최종 정확도 | 90% | 92% (보통 더 좋음) |
| Dropout 의존 | 강함 | 약함 (BN 자체가 약한 정규화) |

**한 줄 요약: 학습이 더 빠르고, 더 안정적이고, 결과도 약간 더 좋아요.**

---

## 학습 모드 vs 평가 모드

⚠️ Dropout처럼, BN도 **모드에 따라 다르게 동작**합니다.

### 학습 모드 (`model.train()`)
- 현재 배치의 평균/분산으로 정규화
- 동시에 **이동 평균(running mean/var)** 도 업데이트

### 평가 모드 (`model.eval()`)
- 학습 중에 누적된 **이동 평균을 사용**
- 현재 배치는 무시 (배치 크기 1이어도 됨)

```python
model.train()   # ← BN이 배치 통계 사용
output = model(x)

model.eval()    # ← BN이 누적 통계 사용
output = model(x)
```

`model.eval()` 안 하면 추론 결과가 매번 다르게 나와요. 정말 자주 만나는 함정.

---

## BatchNorm을 어디에 둬요?

표준은 **`Linear → BatchNorm → Activation`** 순서.

```python
class WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        self.bn = nn.BatchNorm1d(256)    # ← 256은 fc의 출력 차원

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)        # 정규화
        x = torch.relu(x)     # 그리고 활성화
        return x
```

### 왜 활성화 함수 전에?

원논문은 활성화 **이전**, 일부 후속 연구는 **이후**가 더 좋다고 주장. 실무에서는:
- **이전이 표준** (대부분의 경우)
- ResNet 등 일부 구조는 이후

처음에는 **이전**으로 두세요. 큰 차이는 없어요.

---

## BatchNorm 변형들

| 변형 | 정규화 단위 | 사용 |
|------|-----------|------|
| **BatchNorm** | 배치 전체 | MLP, CNN (일반적) |
| **LayerNorm** | 한 샘플의 모든 특성 | **Transformer**, RNN |
| **InstanceNorm** | 한 샘플의 한 채널 | 이미지 스타일 변환 |
| **GroupNorm** | 채널 그룹 단위 | 작은 배치 (BN이 약할 때) |

BatchNorm의 한계: **배치 크기가 1이면 동작 못 해요.** (분산 계산 불가) 이럴 땐 LayerNorm이나 GroupNorm.

```python
# BatchNorm
nn.BatchNorm1d(256)        # MLP
nn.BatchNorm2d(64)         # CNN의 채널 64

# LayerNorm (Transformer 표준)
nn.LayerNorm(256)
```

---

## 코드: BN 효과 직접 확인

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class WithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


# 둘 다 같은 데이터로, 같은 lr로 학습 후 곡선 비교
# (BN이 더 빨리 수렴하고 더 안정적)
```

---

## 정리

```python
# 사용
nn.BatchNorm1d(차원수)    # MLP
nn.BatchNorm2d(채널수)    # CNN

# 위치 (표준)
Linear → BN → ReLU

# 모드 주의
model.train()  # 배치 통계
model.eval()   # 누적 통계
```

**핵심:**
- Internal Covariate Shift 해결
- 학습 빠름, 안정적, 더 큰 학습률 가능
- 평가 모드 꼭 켜기 (`model.eval()`)
- Transformer 같은 곳은 LayerNorm

---

## 자주 묻는 질문

> **Q. BN이 정규화 효과도 있나요? Dropout을 안 써도 되나요?**
>
> 약한 정규화 효과가 있어요. 배치마다 통계가 달라져서 노이즈가 추가되거든요. 다만 Dropout만큼 강력하지는 않아서 **둘 다 같이 쓰는 게 보통** 이에요.

> **Q. 작은 배치에서는 왜 약해져요?**
>
> 배치가 작으면 배치 통계가 부정확해요 (예: 배치 4개로 평균/분산 계산). 그래서 배치 크기 16 이하라면 GroupNorm이나 LayerNorm을 고려.

> **Q. BN이 항상 도움 되나요?**
>
> 거의 항상. 다만 **RNN과 Transformer에서는 LayerNorm이 더 잘 맞아요.** 시퀀스 길이가 다양해서 배치 통계가 의미가 약하거든요.

➡️ 이전: [4.5 더 깊게](05-깊게.md)
➡️ 다음: [4.6 GPU 활용](06-GPU-활용.md)
