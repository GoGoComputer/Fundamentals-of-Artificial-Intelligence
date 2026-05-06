# 4.2 PyTorch 시작하기

> **선행 학습:** PyTorch의 텐서는 **벡터와 행렬**의 연장입니다.  
> [👉 선형대수 입문 (4.1~4.3 벡터와 행렬)](../../부록/수학/04-선형대수-입문.md)  
> 단, 코드부터 배우고 나중에 수학을 깊이 이해해도 괜찮습니다.

---

## PyTorch가 뭔가요?

PyTorch는 **신경망을 만들고 학습시키는 도구**예요. Meta(Facebook)가 만들고 오픈소스로 공개했습니다.

비유하자면, 신경망을 짓는 데 필요한:
- 벽돌과 기둥 (`nn.Linear`, `nn.Conv2d`)
- 모르타르 (자동 미분, autograd)
- 작업 도구 (옵티마이저, 손실 함수)

이런 모든 걸 다 줘요. 우리는 **레고 블록 조립하듯이 신경망을 만들 수 있어요.**

---

## 첫 단계: 텐서 (Tensor)

PyTorch의 가장 기본 단위는 **텐서**예요. **숫자 배열**이라고 보시면 됩니다.

NumPy 배열 아세요? 거의 똑같아요. 차이점은 **GPU에서 돌아간다**는 것 정도.

### 텐서 만들기

```python
import torch

# 직접 만들기
a = torch.tensor([1, 2, 3])
print(a)               # tensor([1, 2, 3])
print(a.shape)         # torch.Size([3])

# 2차원
b = torch.tensor([[1, 2], [3, 4]])
print(b)
# tensor([[1, 2],
#         [3, 4]])
print(b.shape)         # torch.Size([2, 2])

# 영행렬, 일행렬
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
random = torch.rand(2, 2)    # 0~1 사이 무작위
randn = torch.randn(2, 2)    # 정규분포 무작위

# NumPy에서 변환
import numpy as np
np_arr = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_arr)

# Tensor → NumPy
back_to_numpy = torch_tensor.numpy()
```

### 텐서의 자료형

```python
a = torch.tensor([1, 2, 3])
print(a.dtype)         # torch.int64

b = torch.tensor([1.0, 2.0, 3.0])
print(b.dtype)         # torch.float32

# 자료형 지정
c = torch.tensor([1, 2, 3], dtype=torch.float32)

# 변환
d = a.float()          # → float32
e = a.long()           # → int64
```

⚠️ **자주 만나는 에러:** "Expected float but got long" 같은 에러는 자료형이 안 맞을 때 나요. 머신러닝에서는 보통 **float32**를 씁니다.

```python
# 안전한 패턴
X = torch.from_numpy(X_np).float()
y = torch.from_numpy(y_np).long()    # 분류일 때
y = torch.from_numpy(y_np).float()   # 회귀일 때
```

---

## 텐서 연산

NumPy랑 거의 똑같아요.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 산술
print(a + b)           # tensor([5, 7, 9])
print(a * b)           # tensor([4, 10, 18])
print(a.sum())         # tensor(6)
print(a.mean().float()) # tensor(2.0)

# 행렬 곱
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(A @ B)
# tensor([[19, 22],
#         [43, 50]])

# 모양 바꾸기 (reshape)
x = torch.arange(12)
print(x.shape)         # torch.Size([12])
print(x.reshape(3, 4))
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 또는 view (같은 효과, 메모리 공유)
print(x.view(2, 6))
```

---

## GPU 사용

이게 PyTorch의 진짜 매력이에요. **텐서를 GPU로 보낼 수 있어요.**

### GPU 사용 가능한지 확인

```python
import torch

print(torch.cuda.is_available())   # True여야 GPU 사용 가능
print(torch.cuda.device_count())   # GPU 개수
print(torch.cuda.get_device_name(0))   # GPU 이름
```

Colab에서 `런타임 → 런타임 유형 변경 → GPU` 안 하셨으면 False가 나옵니다.

### 텐서를 GPU로 보내기

```python
# device 변수 정의 (관용적인 패턴)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# CPU에서 만들고 GPU로 옮기기
x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)       # cpu

x = x.to(device)
print(x.device)       # cuda:0

# 또는 처음부터 GPU에
y = torch.tensor([1.0, 2.0, 3.0], device=device)
```

⚠️ **함정:** 텐서들이 **같은 device**에 있어야 연산이 됩니다.

```python
a = torch.tensor([1.0])              # CPU
b = torch.tensor([2.0], device='cuda')  # GPU

a + b   # ❌ RuntimeError!
```

해결: 한쪽으로 옮겨주기.

```python
a = a.to(device)
a + b   # OK
```

### GPU 텐서 → NumPy?

NumPy는 CPU에서만 동작해요. GPU 텐서를 NumPy로 바꿀 땐:

```python
gpu_tensor.cpu().numpy()    # CPU로 옮긴 후 numpy로
```

---

## autograd: 자동 미분

이게 PyTorch의 **마법**이에요. 신경망 학습의 핵심 부분.

### 미분이 왜 필요한가요?

신경망 학습은 **loss를 최소화**하는 일이에요. 그러려면 "각 가중치를 어떤 방향으로 바꾸면 loss가 줄어들지"를 알아야 해요. 이게 **그래디언트(gradient, 기울기)** 입니다.

수학적으로는 미분이에요. 손으로 계산하면 어마어마하게 복잡한데, **PyTorch가 자동으로 계산해 줍니다.**

### 사용법

```python
# requires_grad=True 로 표시
x = torch.tensor([2.0], requires_grad=True)

# 어떤 계산을 함
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1

# 미분 계산 (backward)
y.backward()

# x에 대한 미분값
print(x.grad)   # tensor([7.])
# (계산: dy/dx = 2x + 3 = 2*2 + 3 = 7)
```

PyTorch가 알아서 미분을 해 줘요. 신경망의 가중치들을 모두 `requires_grad=True`로 두고, loss에 대해 `backward()`를 호출하면, **모든 가중치의 그래디언트가 자동으로 계산**됩니다.

```python
# 신경망 학습의 본질
loss.backward()                      # 모든 가중치의 그래디언트 자동 계산
optimizer.step()                     # 그래디언트 방향으로 가중치 업데이트
```

이 두 줄이 학습의 핵심이에요.

---

## nn.Module: 신경망의 기본 단위

PyTorch에서 신경망은 **`nn.Module`을 상속받는 클래스**로 만들어요.

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 여기서 층(layer)들을 정의
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        # 여기서 계산 흐름을 정의
        x = self.layer1(x)
        x = torch.relu(x)        # 활성화 함수
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

# 사용
model = MyNetwork()
print(model)
```

```
MyNetwork(
  (layer1): Linear(in_features=784, out_features=256, bias=True)
  (layer2): Linear(in_features=256, out_features=128, bias=True)
  (layer3): Linear(in_features=128, out_features=10, bias=True)
)
```

### 두 가지 핵심 메소드

- **`__init__`**: 사용할 층들을 정의 (생성자)
- **`forward`**: 입력이 들어왔을 때 어떻게 계산할지

이게 PyTorch 신경망의 기본 형태예요. 거의 모든 모델이 이 패턴을 따릅니다.

### 클래스 처음 보시는 분께

[1장 함수](../../01-파이썬-기초/이론/06-함수.md)에서 클래스를 안 다뤘죠. 짧게만 설명드리면:

- `class MyNetwork(nn.Module):` — "MyNetwork라는 새 종류의 객체를 만들 거야. 이건 nn.Module을 기반으로 해."
- `def __init__(self):` — "이 객체가 만들어질 때 일어날 일"
- `super().__init__()` — "부모(nn.Module)의 초기화도 같이 해줘"
- `self.layer1 = ...` — "이 객체 안에 layer1이라는 속성을 둘 거야"
- `model = MyNetwork()` — "MyNetwork 종류의 객체 하나를 만들어"

이 정도만 아셔도 PyTorch 모델 만드시는 데 충분해요.

---

## 모델에 데이터 통과시키기 (forward)

```python
import torch

# 모델 정의 (위에서 만든 것)
model = MyNetwork()

# 가짜 입력 (배치 1개, 784 픽셀)
x = torch.randn(1, 784)

# 모델에 입력
y = model(x)

print(y.shape)    # torch.Size([1, 10])    배치 1, 클래스 10
```

⚠️ `model(x)` 라고 호출했지, `model.forward(x)` 라고 안 한 점에 주목하세요. PyTorch는 `model(x)` 하면 자동으로 `forward(x)`를 부릅니다. 약간 마법 같지만, 이렇게 쓰는 게 관례예요.

---

## 모델 정보 보기

`print(model)` 외에도, 모델의 파라미터 수와 구조를 자세히 보고 싶을 땐 `torchinfo`가 좋아요.

```python
# Colab에 없으면 설치
# !pip install torchinfo

from torchinfo import summary

model = MyNetwork()
summary(model, input_size=(1, 784))
```

```
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
MyNetwork                                [1, 10]                   --
├─Linear: 1-1                            [1, 256]                  200,960
├─Linear: 1-2                            [1, 128]                  32,896
├─Linear: 1-3                            [1, 10]                   1,290
==========================================================================
Total params: 235,146
Trainable params: 235,146
==========================================================================
```

총 **235,146개의 파라미터(가중치)** 가 학습된다는 뜻이에요. 이게 다 자동으로 미분되고 업데이트돼요. 신기하죠.

---

## 정리: PyTorch 첫 단계

```python
import torch
import torch.nn as nn

# 1. 텐서 만들기 (NumPy랑 비슷)
x = torch.tensor([1.0, 2.0, 3.0])

# 2. GPU로 보내기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# 3. 자동 미분 활성화
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)    # 4.0

# 4. 신경망 정의
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

# 5. 모델 사용
model = MyNet().to(device)
output = model(input_tensor)
```

이 다섯 가지가 PyTorch의 80%예요. 다음 글에서 진짜 신경망을 만들어 봅시다.

---

## 자주 묻는 질문

> **Q. NumPy랑 PyTorch 둘 다 있는데 언제 뭘 쓰죠?**
>
> 데이터 전처리는 NumPy/Pandas로, 모델 학습은 PyTorch로. 둘은 서로 변환이 쉬워요. 학습 시작 직전에 NumPy → torch.Tensor로 바꾸시면 됩니다.

> **Q. requires_grad=True를 항상 줘야 하나요?**
>
> 아니요. nn.Linear 같은 층 안의 가중치는 자동으로 requires_grad=True예요. 우리가 직접 텐서로 모델을 만들 때만 명시합니다 (드물어요).

> **Q. .to(device)를 어디에 다 붙여야 해요?**
>
> 모델과 입력 데이터에 둘 다 붙이세요. 같은 device여야 연산이 됩니다.

➡️ 다음: [4.3 첫 신경망 만들기](03-첫-신경망.md)
