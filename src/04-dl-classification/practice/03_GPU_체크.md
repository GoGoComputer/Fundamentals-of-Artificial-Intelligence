# 4장 6절 실습: GPU 환경 체크 + 속도 비교

- 원본 파일: `04-딥러닝-분류/실습/03_GPU_체크.py`
- 동기화 방식: 강의 원본에서 자동 생성

아래는 생략 없이 전체 코드입니다.

```python
"""4장 6절 실습: GPU 환경 체크 + 속도 비교

GPU가 실제로 연결되어 있는지 확인하고,
행렬 곱과 학습 step 기준으로 CPU 대비 속도 차이를 측정합니다.
"""

import torch
import time


# 1단계에서는 현재 파이썬 환경이 GPU를 볼 수 있는지 하드웨어 정보를 출력합니다.
print("=" * 50)
print("GPU 환경 진단")
print("=" * 50)

print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # GPU가 보이면 CUDA 버전, 장치 수, 장치별 메모리 정보를 자세히 출력합니다.
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  메모리: {props.total_memory / 1e9:.2f} GB")
        print(f"  컴퓨트 능력: {props.major}.{props.minor}")
else:
    # GPU가 없으면 Colab에서 켜는 메뉴 경로를 바로 안내합니다.
    print("\n⚠️ GPU를 사용할 수 없습니다.")
    print("Colab에서: 런타임 → 런타임 유형 변경 → GPU")


# 2단계는 가장 단순한 연산(행렬 곱)으로 CPU/GPU 성능 차이를 체감하는 구간입니다.
print("\n" + "=" * 50)
print("CPU vs GPU 속도 비교 (행렬 곱)")
print("=" * 50)

sizes = [128, 512, 2048, 4096]

for size in sizes:
    print(f"\n[{size} x {size} 행렬 곱]")

    # CPU
    # 같은 크기의 랜덤 행렬 두 개를 만들고 10회 평균 시간을 측정합니다.
    A = torch.randn(size, size)
    B = torch.randn(size, size)

    start = time.time()
    for _ in range(10):
        C = A @ B
    cpu_time = (time.time() - start) / 10
    print(f"  CPU: {cpu_time*1000:.2f}ms")

    if torch.cuda.is_available():
        # GPU 측정은 동일 행렬을 cuda 메모리로 복사해 같은 연산을 반복합니다.
        A_gpu = A.cuda()
        B_gpu = B.cuda()

        # 첫 호출은 커널 초기화 비용이 섞이므로 워밍업 1회를 먼저 수행합니다.
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


# 3단계는 실제 학습 패턴(Forward+Backward+Step)으로 장치별 차이를 측정합니다.
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
    # benchmark 함수는 지정 device에서 동일한 모델/배치로 평균 step 시간을 계산합니다.
    device = torch.device(device_name)
    model = TestNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(128, 1024).to(device)
    y = torch.randint(0, 10, (128,)).to(device)

    # 워밍업 단계에서 초기 커널/메모리 할당 비용을 미리 소모합니다.
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    if device_name == 'cuda':
        torch.cuda.synchronize()

    # 본 측정은 50 step 평균으로 계산해 순간 변동을 줄입니다.
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


# 마지막 단계에서는 현재 프로세스의 GPU 메모리 사용량을 점검합니다.
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

    # empty_cache는 캐시된 메모리를 반환해 이후 작업의 OOM 위험을 낮출 때 사용합니다.
    torch.cuda.empty_cache()
    print("\n  empty_cache() 호출")


print("\nGPU 진단 끝!")
```
