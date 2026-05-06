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

[환경 설정](../00-intro/03-setup.md)에서 다뤘죠. 다시 한 번:

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

➡️ 다음: [4.7 현업 체크리스트](07-production-checklist.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **4.6 GPU 활용** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **4.6 GPU 활용**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "4.6 GPU 활용 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **4.6 GPU 활용**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

4.6 GPU 활용는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
