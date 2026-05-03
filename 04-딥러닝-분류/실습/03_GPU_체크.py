"""4장 6절 실습: GPU 환경 체크 + 속도 비교

GPU가 잘 잡혔는지 확인하고, CPU vs GPU 속도 차이를 직접 봅니다.
"""

import torch
import time


# ============================================================
# 1. GPU 환경 확인
# ============================================================
print("=" * 50)
print("GPU 환경 진단")
print("=" * 50)

print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  메모리: {props.total_memory / 1e9:.2f} GB")
        print(f"  컴퓨트 능력: {props.major}.{props.minor}")
else:
    print("\n⚠️ GPU를 사용할 수 없습니다.")
    print("Colab에서: 런타임 → 런타임 유형 변경 → GPU")


# ============================================================
# 2. 속도 비교: 행렬 곱
# ============================================================
print("\n" + "=" * 50)
print("CPU vs GPU 속도 비교 (행렬 곱)")
print("=" * 50)

sizes = [128, 512, 2048, 4096]

for size in sizes:
    print(f"\n[{size} x {size} 행렬 곱]")

    # CPU
    A = torch.randn(size, size)
    B = torch.randn(size, size)

    start = time.time()
    for _ in range(10):
        C = A @ B
    cpu_time = (time.time() - start) / 10
    print(f"  CPU: {cpu_time*1000:.2f}ms")

    if torch.cuda.is_available():
        A_gpu = A.cuda()
        B_gpu = B.cuda()

        # GPU 워밍업 (첫 호출은 항상 느림)
        _ = A_gpu @ B_gpu
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            C = A_gpu @ B_gpu
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10
        print(f"  GPU: {gpu_time*1000:.2f}ms")

        speedup = cpu_time / gpu_time
        print(f"  → GPU가 {speedup:.1f}배 빠름")


# ============================================================
# 3. 신경망 학습 속도 비교
# ============================================================
print("\n" + "=" * 50)
print("신경망 학습 속도 (한 step)")
print("=" * 50)

import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def benchmark(device_name):
    device = torch.device(device_name)
    model = TestNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(128, 1024).to(device)
    y = torch.randint(0, 10, (128,)).to(device)

    # 워밍업
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    if device_name == 'cuda':
        torch.cuda.synchronize()

    # 측정
    start = time.time()
    for _ in range(50):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    if device_name == 'cuda':
        torch.cuda.synchronize()

    duration = (time.time() - start) / 50
    return duration


cpu_step = benchmark('cpu')
print(f"\nCPU 한 step: {cpu_step*1000:.2f}ms")

if torch.cuda.is_available():
    gpu_step = benchmark('cuda')
    print(f"GPU 한 step: {gpu_step*1000:.2f}ms")
    print(f"→ GPU가 {cpu_step/gpu_step:.1f}배 빠름")


# ============================================================
# 4. GPU 메모리 사용량
# ============================================================
if torch.cuda.is_available():
    print("\n" + "=" * 50)
    print("GPU 메모리 사용량")
    print("=" * 50)

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  사용 중: {allocated:.2f} GB")
    print(f"  예약됨:  {reserved:.2f} GB")
    print(f"  총 용량: {total:.2f} GB")
    print(f"  사용률:  {allocated/total*100:.1f}%")

    # 메모리 비우기 (가끔 필요)
    torch.cuda.empty_cache()
    print("\n  empty_cache() 호출")


print("\nGPU 진단 끝!")
