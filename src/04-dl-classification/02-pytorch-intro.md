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

[1장 함수](../01-python-basics/06-functions.md)에서 클래스를 안 다뤘죠. 짧게만 설명드리면:

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

➡️ 다음: [4.3 첫 신경망 만들기](03-first-neural-network.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **4.2 PyTorch 시작하기** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **4.2 PyTorch 시작하기**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

---

## 자주 하는 오해와 바로잡기

### 1) "이론만 이해하면 실습은 저절로 된다"
아닙니다. 이론 이해와 코드 실행 능력은 별개의 근육입니다. 둘을 같이 써야 합니다.

### 2) "예제 코드가 돌아갔으니 완전히 이해했다"
동작 확인은 시작일 뿐입니다. 파라미터를 바꿨을 때 결과가 어떻게 달라지는지까지 확인해야 이해가 완성됩니다.

### 3) "좋은 모델은 항상 하나다"
데이터 특성과 제약(시간, 메모리, 응답 속도)에 따라 최선의 선택은 달라집니다.

### 4) "지표 숫자만 높으면 끝이다"
지표는 해석이 필요합니다. 어떤 데이터에서, 어떤 분포에서, 어떤 기준으로 얻은 점수인지 함께 보아야 합니다.

### 5) "지금은 초반이니까 배포는 나중에 생각해도 된다"
초반부터 재현성과 저장 구조를 습관화해야 나중에 배포 단계에서 무너지지 않습니다.

---

## 실전 연결 시나리오

아래는 이 단원 내용을 실제 업무로 연결할 때 자주 쓰는 흐름입니다.

1. 문제를 한 문장으로 정의한다.  
예: "4.2 PyTorch 시작하기 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

2. 성공 기준을 지표로 정한다.  
예: 정확도/RMSE/F1/지연시간 같은 운영 가능한 숫자로 정의.

3. 가장 단순한 베이스라인을 만든다.  
복잡한 모델보다 먼저, 빠르게 재현 가능한 기본 모델을 세웁니다.

4. 한 번에 하나씩 개선한다.  
전처리, 모델, 튜닝, 임계값, 배치 크기 등은 동시에 바꾸지 않고 순차적으로 바꿉니다.

5. 결과를 로그와 함께 저장한다.  
실험 이름, 파라미터, 점수, 데이터 버전을 함께 남겨야 재현이 됩니다.

6. 실패 케이스를 분석한다.  
점수 평균보다 오분류/고오차 샘플을 분석할 때 개선 아이디어가 나옵니다.

7. 배포 가능 형태로 정리한다.  
모델 파일, 메타데이터, 입력 스키마, 테스트 스크립트까지 묶습니다.

---

## 복습 체크리스트

다음 질문에 답할 수 있으면 이 단원을 실전 수준으로 이해한 것입니다.

- 이 단원의 핵심 개념을 중학생에게 설명하듯 말할 수 있는가?
- 코드의 각 블록이 왜 필요한지 한 줄씩 설명할 수 있는가?
- 이 단원에서 가장 자주 틀리는 지점을 알고 있는가?
- 지표를 볼 때 어떤 함정이 있는지 알고 있는가?
- 베이스라인과 개선 모델의 차이를 표로 정리할 수 있는가?
- 재현 가능하게 실험을 저장하는 습관이 있는가?
- 실패 사례를 보고 다음 실험 가설을 만들 수 있는가?
- 이 단원 내용을 다음 장과 어떻게 연결할지 설명할 수 있는가?

---

## 확장 연습 과제

### 과제 A: 파라미터 감각 만들기
현재 예제에서 핵심 파라미터 1~2개만 바꿔 5회 실험하고, 결과를 표로 정리해 보세요.

### 과제 B: 실패 사례 분석
오분류 또는 오차가 큰 샘플 20개를 모아 공통 패턴을 찾아 보세요.

### 과제 C: 설명 가능 리포트 작성
"무엇을 했고, 왜 했고, 결과가 어땠고, 다음엔 무엇을 할지"를 1페이지로 정리해 보세요.

이 3개를 직접 해 보면 **4.2 PyTorch 시작하기**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

4.2 PyTorch 시작하기는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
