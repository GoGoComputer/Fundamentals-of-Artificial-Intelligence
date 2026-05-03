"""4장 1~2절 실습: PyTorch 텐서와 자동 미분 입문

PyTorch의 기본 도구들을 한 줄씩 손에 익히세요.
"""

import torch
import numpy as np


# ============================================================
# 1. 텐서 만들기
# ============================================================
print("=" * 50)
print("1. 텐서 만들기")
print("=" * 50)

# 직접
a = torch.tensor([1, 2, 3])
print(f"a = {a}, shape = {a.shape}, dtype = {a.dtype}")

# 2차원
b = torch.tensor([[1, 2], [3, 4]])
print(f"b = \n{b}")
print(f"b shape = {b.shape}")

# 영행렬, 일행렬, 랜덤
print(f"\nzeros(3,4):\n{torch.zeros(3, 4)}")
print(f"ones(2,3):\n{torch.ones(2, 3)}")
print(f"randn(2,2):\n{torch.randn(2, 2)}")

# NumPy와 변환
np_arr = np.array([1, 2, 3])
t = torch.from_numpy(np_arr)
back = t.numpy()
print(f"\nNumPy → torch: {t}")
print(f"torch → NumPy: {back}")


# ============================================================
# 2. 텐서 자료형
# ============================================================
print("\n" + "=" * 50)
print("2. 자료형")
print("=" * 50)

a = torch.tensor([1, 2, 3])
print(f"기본 정수: {a.dtype}")           # int64

b = torch.tensor([1.0, 2.0, 3.0])
print(f"기본 실수: {b.dtype}")           # float32

# 변환
c = a.float()
d = b.long()
print(f"long → float: {c.dtype}")
print(f"float → long: {d.dtype}")


# ============================================================
# 3. 텐서 연산 (NumPy랑 비슷)
# ============================================================
print("\n" + "=" * 50)
print("3. 연산")
print("=" * 50)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a.sum() = {a.sum()}")
print(f"a.float().mean() = {a.float().mean()}")

# 행렬 곱
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(f"\n행렬 곱:\n{A @ B}")

# 모양 바꾸기
x = torch.arange(12)
print(f"\n원래: {x}")
print(f"reshape(3, 4):\n{x.reshape(3, 4)}")
print(f"view(2, 6):\n{x.view(2, 6)}")


# ============================================================
# 4. GPU 사용
# ============================================================
print("\n" + "=" * 50)
print("4. GPU")
print("=" * 50)

print(f"CUDA 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

# device 정의 (관용적)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용할 device: {device}")

# 텐서를 GPU로
x = torch.tensor([1.0, 2.0, 3.0])
print(f"옮기기 전 device: {x.device}")
x = x.to(device)
print(f"옮긴 후 device: {x.device}")


# ============================================================
# 5. 자동 미분 (autograd)
# ============================================================
print("\n" + "=" * 50)
print("5. 자동 미분 (PyTorch의 마법)")
print("=" * 50)

# 단순한 예: y = x²
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2
print(f"x = {x.item()}, y = x² = {y.item()}")

y.backward()
print(f"dy/dx = 2x = {x.grad.item()}")    # 6.0


# 좀 더 복잡한 예
print("\n[좀 더 복잡한 함수]")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x ** 2 + x + 5
print(f"y = x³ + 2x² + x + 5 = {y.item()}")

y.backward()
print(f"dy/dx = 3x² + 4x + 1 = {x.grad.item()}")    # 21.0


# 신경망에서는 이런 식으로
print("\n[신경망에서의 활용 (개념)]")
weights = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
input_data = torch.tensor([1.0, 2.0, 3.0])

# forward
output = (weights * input_data).sum()
target = torch.tensor(2.0)
loss = (output - target) ** 2

print(f"output = {output.item():.4f}")
print(f"loss = {loss.item():.4f}")

# backward
loss.backward()
print(f"\nweights.grad = {weights.grad}")
print("→ 이 그래디언트의 반대 방향으로 가중치를 업데이트하면 loss가 줄어듭니다")


# ============================================================
# 6. 첫 nn.Module
# ============================================================
print("\n" + "=" * 50)
print("6. 첫 신경망 (nn.Module)")
print("=" * 50)

import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet().to(device)
print(model)

# 가짜 입력 통과
x = torch.randn(3, 10).to(device)    # 배치 3, 특성 10
y = model(x)
print(f"\n입력 모양: {x.shape}")
print(f"출력 모양: {y.shape}")

# 파라미터 수
n_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터 수: {n_params}")
# (10*5+5) + (5*2+2) = 67


print("\nPyTorch 입문 끝! 이제 진짜 신경망을 만들 준비가 됐어요.")
