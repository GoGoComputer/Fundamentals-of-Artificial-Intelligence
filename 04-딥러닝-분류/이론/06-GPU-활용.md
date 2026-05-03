# 4.6 GPU 활용

## "GPU가 그렇게 빨라요?"

네, 진짜 빨라요. 같은 모델/같은 데이터로 비교하면:

| 하드웨어 | 1 epoch 시간 (MNIST) |
|---------|---------------------|
| CPU (M1 Mac) | ~30초 |
| Colab GPU (T4) | ~3초 |
| RTX 4090 | ~0.5초 |

**10~100배 차이**가 나요. 큰 모델일수록 차이가 더 커집니다.

이 글에서는 GPU를 잘 쓰는 법을 정리합니다.

---

## CPU vs GPU — 왜 차이가 큰가요?

### CPU
- 4~16개의 코어
- 코어 하나하나가 똑똑함 (복잡한 명령어 처리)
- **순차적 처리**에 강함

### GPU
- 수천 개의 코어
- 각 코어는 단순함
- **병렬 처리**에 강함

신경망 연산은 **행렬 곱**이 핵심이에요. 행렬 곱은 수많은 단순 연산을 동시에 할 수 있어요. 이게 GPU의 강점과 정확히 맞아요.

```
신경망: 행렬 곱 X 행렬 곱 X 행렬 곱 ...
        ↓
        GPU의 수천 코어가 한꺼번에 처리
```

---

## Colab에서 GPU 켜기

[환경 설정](../../00-시작하기-전에/03-환경-설정.md)에서 다뤘죠. 다시 한 번:

1. 메뉴: **`런타임 → 런타임 유형 변경`**
2. 하드웨어 가속기: **`GPU`** (또는 `T4 GPU`)
3. 저장

확인:

```python
import torch
print(torch.cuda.is_available())   # True여야 함
print(torch.cuda.get_device_name(0))   # 'Tesla T4' 등
```

False면 위 단계를 다시 해 보세요.

---

## PyTorch 코드에 GPU 적용

코드 한 줄씩 어디에 `.to(device)` 를 붙여야 하는지 정리합니다.

### 1. device 정의 (제일 먼저)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

이 한 줄이 코드 호환성을 보장해요. GPU 있으면 GPU, 없으면 CPU.

### 2. 모델을 GPU로

```python
model = MyModel()
model = model.to(device)
```

또는 한 줄로:

```python
model = MyModel().to(device)
```

### 3. 데이터를 GPU로 (학습 루프 안에서)

```python
for images, labels in train_loader:
    images = images.to(device)    # GPU로
    labels = labels.to(device)    # GPU로
    
    # ... 학습 코드 ...
```

⚠️ DataLoader는 데이터를 CPU에 갖고 있어요. **매 배치마다 GPU로 옮겨야** 해요.

### 함정: device 일치

모델과 데이터가 같은 device에 있어야 해요. 안 그러면:

```
RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, cuda:0 and cpu!
```

이런 에러가 떠요. 자주 만나는 에러니 익숙해지세요.

---

## GPU 사용량 체크

학습 중 GPU 사용률을 보고 싶으시면:

```python
# Colab 셀에서
!nvidia-smi
```

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10                |
+-----------------------------------------------------------------------------+
| 0  Tesla T4         On    | 00000000:00:04.0 Off |                    0 |
| N/A   65C    P0    72W /  70W |   3215MiB / 15360MiB |     85%      Default |
+-----------------------------------------------------------------------------+
```

- **GPU 사용률 (85%)**: 높을수록 GPU를 잘 쓰고 있는 것
- **메모리 (3215/15360 MB)**: 모델 + 데이터 + 그래디언트가 차지

학습 중인데 사용률이 10% 같이 낮으면, 데이터 로딩이 병목이에요. (`num_workers` 옵션 늘리세요)

---

## 데이터 로딩 가속: num_workers

DataLoader에서 데이터를 가져올 때 여러 프로세스를 쓸 수 있어요.

```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # 4개 프로세스로 병렬 로딩
    pin_memory=True,      # GPU 전송 빨라짐
)
```

- `num_workers=0`: 메인 프로세스만 (기본). GPU가 데이터 기다림
- `num_workers=4`: 4개 워커가 병렬로 다음 배치 준비
- `pin_memory=True`: 호스트 메모리를 고정해서 GPU 전송 빠르게

⚠️ Colab/Jupyter에서 num_workers > 0이 가끔 문제가 됩니다. 안 되면 0으로 두세요.

---

## Mixed Precision Training (FP16)

요즘 신경망은 보통 FP32(32비트 부동소수)로 계산해요. 이걸 **FP16(반정밀도)** 로 바꾸면:

- 메모리 절반
- 계산 속도 2~3배

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    with autocast():
        outputs = model(images)
        loss = loss_fn(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

복잡해 보이지만, 5줄만 추가하면 학습이 빨라져요. RTX 30/40 시리즈, T4, V100 등에서 효과적.

---

## GPU 메모리 부족 (OOM) 대처법

큰 모델이나 큰 배치를 쓰면 자주 마주치는 에러:

```
RuntimeError: CUDA out of memory.
```

### 해결 방법

1. **batch_size 줄이기** — 가장 쉬운 방법
2. **모델 크기 줄이기**
3. **`torch.cuda.empty_cache()`** — 사용하지 않는 메모리 비움
4. **Gradient Accumulation** — 작은 배치로 큰 배치 효과

### Gradient Accumulation

```python
accumulation_steps = 4   # 배치 16 × 4 = 64 효과

optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    loss = loss_fn(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

배치 16개씩 4번 누적해서 배치 64개의 효과. 메모리 적게 쓰면서 큰 배치 효과를 얻을 수 있어요.

---

## 여러 GPU (DataParallel)

GPU가 여러 대 있으면 모델을 복제해서 병렬 학습.

```python
# 가장 단순한 방법 (옛날 방식)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

요즘은 더 효율적인 `DistributedDataParallel`을 쓰지만 코드가 복잡해요. Colab은 GPU 1대라 신경 안 쓰셔도 됩니다.

---

## CPU에서 추론 (배포 시)

학습은 GPU에서, 배포는 CPU에서 하는 경우가 많아요. 모델을 옮길 수 있어요.

```python
# GPU에서 학습
model = model.cuda()
# ... 학습 ...

# CPU로 옮겨서 저장
model = model.cpu()
torch.save(model.state_dict(), 'model.pth')

# 다른 곳에서 CPU로 불러서 사용
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 예측
with torch.no_grad():
    output = model(input_data)
```

---

## 학습 속도 최적화 정리

다음 순서로 시도해 보세요.

```
1. .to(device) 다 했나 (모델, 데이터)
   → 안 했으면 즉시 적용

2. num_workers + pin_memory
   → 데이터 로딩이 병목인 경우

3. 배치 사이즈 늘리기
   → 메모리 여유 있으면

4. Mixed Precision (autocast)
   → 최신 GPU에서 효과적

5. 모델/데이터 크기 줄이기
   → 모든 게 다 안 되면 마지막
```

---

## 정리

```python
# 1. device 정의 (코드 시작 부분에)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 모델 to device
model = MyModel().to(device)

# 3. 데이터 로더 최적화
loader = DataLoader(
    dataset, batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True,
)

# 4. 학습 루프 안에서 데이터 to device
for x, y in loader:
    x, y = x.to(device), y.to(device)
    # ...

# 5. (선택) Mixed Precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(x)
    loss = loss_fn(output, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**핵심: device 정의 → 모델 to → 데이터 to.** 이 세 가지만 잘 챙기면 GPU의 90%를 쓸 수 있어요.

➡️ 다음: [4.7 현업 체크리스트](07-현업-체크리스트.md)
