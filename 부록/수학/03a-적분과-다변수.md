# 3a. 적분과 다변수 미적분

> [3장](03-미분-입문.md)에서 미분을 봤죠. 여기서는 **적분과 다변수 미적분**을 채웁니다.
> 적분은 ML에서 직접 쓰는 일은 적지만, 확률 식을 읽기 위해 필요해요.

---

## 3a.1 적분 (Integration) — 미분의 반대

### 직관

미분이 "함수의 기울기"라면, **적분은 함수 아래의 면적**.

```
       y
       │     ___
       │   ╱│  │╲
       │ ╱  │  │  ╲    ← f(x) 그래프
       │╱   │██│   ╲
       └────┼██┼─────→ x
            a   b
            
        ↑
     a부터 b까지 면적 = ∫ₐᵇ f(x) dx
```

### 표기

```
∫ₐᵇ f(x) dx
```

읽으면: **"a부터 b까지 f(x) dx의 적분."**

`dx`는 "x 방향의 작은 조각" 이라는 뜻이에요.

### 부정적분 vs 정적분

#### 부정적분 (Indefinite Integral)

```
∫ f(x) dx = F(x) + C
```

`F(x)`는 미분하면 `f(x)`가 되는 함수. `C`는 적분 상수.

예시:
```
∫ x dx = x²/2 + C    (확인: (x²/2)' = x ✓)
∫ x² dx = x³/3 + C
∫ eˣ dx = eˣ + C
∫ (1/x) dx = log|x| + C
```

#### 정적분 (Definite Integral) — 진짜 면적

```
∫ₐᵇ f(x) dx = F(b) - F(a)
```

부정적분의 끝값 - 시작값.

예시: `0`부터 `1`까지 `x²` 의 적분?

```
F(x) = x³/3
∫₀¹ x² dx = F(1) - F(0) = 1/3 - 0 = 1/3
```

면적이 `1/3`.

### 미적분의 기본 정리

```
∫ₐᵇ f'(x) dx = f(b) - f(a)
```

**미분과 적분은 서로의 역연산.** 이게 미적분 전체를 관통하는 가장 중요한 정리예요.

### ML에서 적분이 어디?

#### 1. 연속 확률 분포

확률 밀도 함수 `p(x)`의 면적이 확률.

```
P(a < X < b) = ∫ₐᵇ p(x) dx
```

전체 면적은 1:
```
∫_{-∞}^{∞} p(x) dx = 1
```

#### 2. 기댓값 (연속형)

```
E[X] = ∫ x × p(x) dx
```

이산형의 시그마(Σ)가 연속형에서는 적분(∫)으로 변해요.

#### 3. KL Divergence, Cross-Entropy 등 정보 이론 식

```
H(p) = -∫ p(x) log p(x) dx
```

다 적분 들어 있어요. 그래도 직접 풀 일은 별로 없고, **읽을 수만 있으면 됩니다.**

### 다행히 코드에서는?

PyTorch나 TensorFlow는 **이산화**해서 처리해요. 진짜 적분 안 해도 돼요.

```python
# 적분 = 작은 조각들의 합으로 근사
total = 0
for x in np.linspace(0, 1, 1000):
    total += f(x) * (1/1000)
# ≈ ∫₀¹ f(x) dx
```

수치 적분은 보통 라이브러리가 처리.

```python
from scipy.integrate import quad
result, _ = quad(lambda x: x**2, 0, 1)
print(result)    # 0.333...
```

---

## 3a.2 다변수 미분 깊이

[3장](03-미분-입문.md)에서 편미분을 봤죠. 더 깊게 들어갑시다.

### 그래디언트 (Gradient) — 다시 정확히

`f(x, y, z, ...)` 의 그래디언트는 모든 편미분의 모음 — **벡터**.

```
∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z, ...)
```

### 그래디언트의 의미

> **그래디언트의 방향 = 함수가 가장 빠르게 증가하는 방향.**
> **그래디언트의 크기 = 그 방향의 변화율.**

신경망 학습에서 **그래디언트의 반대 방향**으로 가는 이유.

### 예시

`f(x, y) = x² + y²` 의 그래디언트:

```
∂f/∂x = 2x
∂f/∂y = 2y

∇f = (2x, 2y)
```

`(1, 0)` 점에서: `∇f = (2, 0)` → x 방향으로 가장 빠르게 증가.

---

## 3a.3 Jacobian 행렬

벡터를 입력으로, 벡터를 출력으로 하는 함수의 미분.

```
f: R^n → R^m
f(x₁, x₂, ..., xₙ) = (f₁, f₂, ..., fₘ)

Jacobian J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
             [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
             [   ⋮         ⋮       ⋱      ⋮  ]
             [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ]
```

m×n 행렬이에요.

### ML에서?

신경망 한 층의 Jacobian이 핵심. 입력 벡터 → 출력 벡터의 변화율.

역전파(backprop)가 사실은 **Jacobian의 곱**의 연속이에요.

### PyTorch에서

```python
# 한 입력에 대한 전체 Jacobian
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([x[0]**2, x[1]**3, x[0]*x[1]])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(J)
# [[2, 0],
#  [0, 12],
#  [2, 1]]
```

직접 손으로 안 해도 PyTorch가 해줘요. 다만 **개념을 알아야** 코드를 이해할 수 있어요.

---

## 3a.4 Hessian 행렬 — 2차 미분

함수의 2차 편미분 모음. 정사각행렬.

```
H = [∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ]
    [∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ]
    [   ⋮          ⋮          ⋱       ⋮      ]
    [∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²  ]
```

### 직관

함수의 **곡률**을 알려줘요. 1차 미분(그래디언트)은 기울기, 2차 미분(헤시안)은 그 기울기가 어떻게 변하는지.

### 예시

`f(x, y) = x² + 2xy + 3y²` 의 Hessian:

```
∂²f/∂x² = 2
∂²f/∂y² = 6
∂²f/∂x∂y = ∂²f/∂y∂x = 2

H = [2  2]
    [2  6]
```

### ML에서?

#### Newton's Method (2차 최적화)

```
w_new = w_old - H⁻¹ × ∇L
                ↑
           Hessian의 역행렬
```

이론적으로 더 빠름. 다만 **Hessian이 너무 커서 (n×n)** 큰 신경망엔 안 씀. 대신 근사 방법 (BFGS, L-BFGS).

#### 손실 곡면 분석

평탄한 minimum vs 가파른 minimum 등.

---

## 3a.5 다변수 함수의 최댓값/최솟값

### 1차 조건 (Necessary Condition)

극값에서는 모든 편미분 = 0.

```
∇f = 0  →  극값 후보
```

### 2차 조건 (Sufficient Condition)

Hessian으로 판별.

| Hessian의 성격 | 의미 |
|---------------|------|
| **양정치 (positive definite)** | 극소 |
| **음정치 (negative definite)** | 극대 |
| **부정 (indefinite)** | saddle point (안장점) |

### Saddle Point — ML의 골치

신경망 학습에서 자주 만나는 문제. 그래디언트가 0이지만 minimum이 아닌 곳.

```
        z
        │
        │       ↗ ↘  ← x 방향으로 증가
        │     /     \
        │    ●  ←  여기서 멈춤 (그래디언트 0)
        │     \     /
        │       ↗ ↘  ← y 방향으로 감소
        └─────────────→
```

좌우로는 위로, 앞뒤로는 아래로 가는 모양 (말 안장처럼). 그래디언트 0이지만 진짜 최저점이 아니에요.

해결: Adam 같은 **모멘텀 옵티마이저**가 안장점을 잘 빠져나옴.

---

## 3a.6 체인룰 (다시) — 다변수

### 단순한 체인룰

`y = f(g(x))` → `dy/dx = f'(g(x)) × g'(x)`

### 다변수 체인룰

`z = f(x, y)` 이고 `x = g(t), y = h(t)` 라면?

```
dz/dt = ∂f/∂x × dx/dt + ∂f/∂y × dy/dt
```

각 변수 경로의 합.

### 신경망의 역전파 (다시)

```
손실 L
   ↑
y₃ = activation(W₃ h₂ + b₃)
   ↑
h₂ = activation(W₂ h₁ + b₂)
   ↑
h₁ = activation(W₁ x + b₁)
```

`∂L/∂W₁` 을 구하려면 다변수 체인룰을 깊이만큼 적용:

```
∂L/∂W₁ = ∂L/∂y₃ × ∂y₃/∂h₂ × ∂h₂/∂h₁ × ∂h₁/∂W₁
         └─────────── 체인룰 ────────────┘
```

이게 역전파의 정체. PyTorch가 자동으로.

---

## 3a.7 Vanishing/Exploding Gradient

체인룰의 부작용. **체인이 길어질수록 그래디언트가 0이 되거나 폭발.**

### 직관

```
∂L/∂W₁ = (각 층의 미분)들의 곱

각 층의 미분이 0.5라면:
  10층: 0.5^10 ≈ 0.001  (vanishing — 거의 0)
  100층: 0.5^100 ≈ 매우 작음  (학습 불가)

각 층의 미분이 2라면:
  10층: 2^10 = 1024
  100층: 2^100 = 천문학적 (exploding — NaN)
```

### 해결책

- **활성화 함수**: ReLU (sigmoid의 vanishing 문제 해결)
- **가중치 초기화**: Xavier, He 초기화
- **Batch Normalization**: 분포 안정화
- **Skip connection**: ResNet의 핵심
- **Gradient Clipping**: exploding 방지

다 본문에서 다룬 내용이에요. 이제 **왜 필요한지** 수학적으로 보이시죠.

---

## 3a.8 정리

```python
import numpy as np
from scipy.integrate import quad

# 정적분
result, _ = quad(lambda x: x**2, 0, 1)
print(result)    # 0.333...

# 그래디언트
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = x[0]**2 + x[1]**3
f.backward()
print(x.grad)    # [2., 12.] = ∇f

# Jacobian
from torch.autograd.functional import jacobian
def func(x):
    return torch.stack([x[0]**2, x[1]**3])
J = jacobian(func, torch.tensor([1.0, 2.0]))

# Hessian
from torch.autograd.functional import hessian
H = hessian(lambda x: (x[0]**2 + x[1]**2), torch.tensor([1.0, 2.0]))
```

### ML 식 다시 읽기

```
∫ p(x) dx = 1
```
"확률 밀도 함수 p(x)의 적분 = 1." (확률의 합)

```
E[X] = ∫ x p(x) dx
```
"X의 기댓값 = x × p(x)의 적분." (기댓값 정의)

```
∇L = (∂L/∂w₁, ..., ∂L/∂wₙ)
```
"L의 그래디언트 = 모든 편미분의 모음."

```
J = [∂fᵢ/∂xⱼ]
```
"Jacobian의 (i,j) 원소 = i번째 출력의 j번째 입력에 대한 편미분."

---

## 자가 진단

1. `∫ x³ dx = ?`
2. `∫₀² 2x dx = ?`
3. `f(x, y) = x²y + y²` 의 그래디언트는?
4. 신경망에서 그래디언트가 vanishing 되는 이유 한 줄로?
5. PyTorch에서 그래디언트 자동 계산 함수는?

<details>
<summary>정답</summary>

1. `x⁴/4 + C`
2. `[x²]₀² = 4 - 0 = 4`
3. `∇f = (2xy, x² + 2y)`
4. "각 층의 미분이 작은 값이면 곱이 거의 0이 됨 (체인룰의 부작용)"
5. `loss.backward()` 또는 `torch.autograd.grad()`

</details>

➡️ 다음: [4. 선형대수 입문](04-선형대수-입문.md) (또는 [4a. 선형대수 심화](04a-선형대수-심화.md))
