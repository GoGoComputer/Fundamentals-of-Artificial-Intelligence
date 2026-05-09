# 7.2 CNN 입문 — 이미지를 보는 신경망

## 들어가며

지금까지 다뤘던 MLP(다층 퍼셉트론)에 이미지를 넣어본 적 있으신가요? `28×28` MNIST 정도는 그럭저럭 학습됩니다. 그런데 `224×224` 컬러 이미지를 넣는 순간 첫 층의 가중치만 1억 개를 훌쩍 넘어버립니다.

```
입력: 224 × 224 × 3 = 150,528 픽셀
첫 은닉층(1024 뉴런)과 완전연결: 150,528 × 1024 ≈ 1.5억 개 가중치
```

게다가 더 심각한 문제가 있습니다. 같은 고양이 사진이어도 고양이가 왼쪽에 있을 때와 오른쪽에 있을 때, MLP는 이를 **완전히 다른 입력**으로 봅니다. 이미지에서 "위치가 달라도 같은 고양이"라는 사실을 배우려면 모든 위치에 대한 학습 데이터가 필요한 셈이죠.

CNN(Convolutional Neural Network)은 이 두 문제를 한 번에 해결합니다.

> 🧮 **수학 보충** — 이번 장의 합성곱 연산은 [부록/수학/03-선형대수.md](../../부록/수학/03-선형대수.md)의 행렬 연산과 [04a-편미분-그래디언트.md](../../부록/수학/04a-편미분-그래디언트.md)의 함수 합성을 알면 더 깊이 이해됩니다. 일단은 직관 위주로 따라오시면 됩니다.

---

## 1. 합성곱(Convolution) — 작은 창으로 이미지 보기

### 1.1 직관: 손전등으로 어두운 방 둘러보기

어두운 방에 들어가서 손전등으로 한 군데씩 비춰본다고 상상해보세요. 작은 영역만 보이지만, 손전등을 천천히 옮기면서 방 전체를 파악할 수 있습니다.

CNN은 이미지를 정확히 이렇게 봅니다.

- **손전등** = 필터(filter, 또는 커널 kernel) — 보통 `3×3`이나 `5×5` 크기
- **비추는 행위** = 합성곱 연산
- **방 전체** = 입력 이미지

같은 손전등(같은 가중치)을 이미지 전체에 슬라이딩시키므로, 가중치 수가 폭발하지 않습니다. `3×3` 필터는 가중치가 단 9개입니다.

### 1.2 구체적인 계산

`5×5` 입력에 `3×3` 필터를 적용해봅니다.

```
입력 이미지              필터(에지 검출용)
1 2 3 0 1               -1 -1 -1
0 1 2 3 0                0  0  0
1 0 1 2 3               +1 +1 +1
2 1 0 1 2
3 2 1 0 1
```

필터를 왼쪽 위 모서리에 올리고 원소별 곱셈 후 모두 더합니다.

```
(1×-1) + (2×-1) + (3×-1)
+ (0×0) + (1×0) + (2×0)
+ (1×1) + (0×1) + (1×1)
= -6 + 0 + 2 = -4
```

이 값이 출력의 `(0, 0)` 위치에 들어갑니다. 필터를 한 칸씩 옮기며 같은 계산을 반복합니다.

### 1.3 stride, padding, 출력 크기

세 가지 하이퍼파라미터가 출력 크기를 결정합니다.

| 용어 | 의미 | 예시 |
|------|------|------|
| stride | 필터를 옮기는 간격 | `stride=2`면 두 칸씩 점프 |
| padding | 가장자리에 0을 덧대는 두께 | `padding=1`이면 사방에 한 줄씩 |
| kernel_size | 필터 크기 | `3×3`, `5×5`, `7×7`이 흔함 |

출력 크기 공식입니다.

```
출력 = (입력 + 2×padding − kernel) / stride + 1
```

`224×224` 입력에 `kernel=3, padding=1, stride=1`이면 `(224 + 2 − 3)/1 + 1 = 224`. 입력 크기를 그대로 유지합니다. 이 조합이 가장 흔하게 쓰입니다.

### 1.4 채널과 다중 필터

컬러 이미지는 RGB 3채널입니다. 그래서 필터도 `3×3×3` 모양이 됩니다.

여러 종류의 패턴을 잡고 싶으니, 보통 한 층에서 32개, 64개, 128개씩 필터를 씁니다. 각 필터가 서로 다른 패턴(가로 에지, 세로 에지, 곡선, 색깔 변화...)을 학습합니다.

```python
import torch
import torch.nn as nn

# 입력 채널 3(RGB), 출력 채널 32(필터 32개), 필터 크기 3×3
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

x = torch.randn(1, 3, 224, 224)  # (배치, 채널, 높이, 너비)
out = conv(x)
print(out.shape)  # torch.Size([1, 32, 224, 224])
```

이 한 층의 가중치 수는 `3 × 3 × 3 × 32 + 32(편향) = 896개`. MLP의 1.5억과 비교하면 16만 배 적습니다.

---

## 2. 풀링(Pooling) — 해상도 줄이기

합성곱만 쌓으면 해상도가 224 그대로라 계산이 무겁고, 멀리 떨어진 픽셀끼리의 관계를 파악하기 어렵습니다. 그래서 중간중간 **풀링**으로 크기를 줄입니다.

### 2.1 Max Pooling vs Average Pooling

`2×2` 영역을 한 값으로 압축하는 방식이 두 가지 있습니다.

```
2×2 영역
1 3
2 8

Max Pooling:     8 (최댓값)
Average Pooling: 3.5 (평균)
```

실전에서는 거의 Max Pooling을 씁니다. "이 영역에 강한 신호가 있었느냐"가 보통 분류에 더 중요한 정보입니다.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 32, 224, 224)
out = pool(x)
print(out.shape)  # torch.Size([1, 32, 112, 112]) — 절반으로 줄어듦
```

### 2.2 Global Average Pooling

최근 모델은 마지막에 `nn.AdaptiveAvgPool2d(1)`을 자주 씁니다. 어떤 크기의 feature map이 들어와도 `1×1`로 만들어버려서, 입력 이미지 크기에 덜 민감한 모델을 만들 수 있습니다.

---

## 3. CNN의 전형적인 구조

대부분의 분류용 CNN은 이런 패턴을 따릅니다.

```
[Conv → ReLU → Conv → ReLU → Pool]   ← 블록 1: 저수준 특징(에지)
[Conv → ReLU → Conv → ReLU → Pool]   ← 블록 2: 중수준 특징(텍스처)
[Conv → ReLU → Conv → ReLU → Pool]   ← 블록 3: 고수준 특징(부분)
[Global Avg Pool]
[Linear → 출력]                       ← 분류기
```

층이 깊어질수록 추상도가 올라갑니다. 첫 층은 "여기 가로 에지가 있다"를 잡고, 마지막 층 근처에서는 "여기 고양이의 귀가 있다" 수준의 정보를 잡습니다. 이걸 **계층적 특징 학습(hierarchical feature learning)**이라고 부릅니다.

### 3.1 작은 CNN 직접 만들어보기

CIFAR-10(10개 클래스, 32×32 컬러 이미지)을 분류하는 간단한 CNN입니다.

```python
import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 블록 1: 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 블록 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 블록 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # 4×4 → 1×1
            nn.Flatten(),                 # (B, 128, 1, 1) → (B, 128)
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SmallCNN()
x = torch.randn(2, 3, 32, 32)
print(model(x).shape)  # torch.Size([2, 10])

total_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {total_params:,}")  # 약 30만 개
```

채널 수를 32 → 64 → 128로 두 배씩 늘리고, 공간 해상도를 풀링으로 절반씩 줄이는 패턴이 반복됩니다. 이 "공간은 줄이고 채널은 늘리기"는 CNN의 거의 모든 곳에서 보이는 관용구입니다.

---

## 4. 유명한 CNN 아키텍처 흐름 한 번에 보기

CNN의 역사는 "어떻게 더 깊게 쌓느냐"의 역사입니다.

| 연도 | 모델 | 깊이 | 핵심 아이디어 |
|------|------|------|---------------|
| 1998 | LeNet-5 | 5층 | 최초의 실용 CNN, 우편번호 인식 |
| 2012 | AlexNet | 8층 | ImageNet 우승, ReLU와 GPU 학습 대중화 |
| 2014 | VGG | 16/19층 | 모든 필터를 `3×3`으로 단순화 |
| 2014 | GoogLeNet | 22층 | Inception 모듈, 1×1 합성곱 |
| 2015 | ResNet | 152층 | **잔차 연결**(skip connection)로 100층 돌파 |
| 2017 | DenseNet | 121층 | 모든 이전 층을 다음 층에 연결 |
| 2019 | EfficientNet | 가변 | 깊이/너비/해상도 균형 잡힌 스케일링 |
| 2021 | ConvNeXt | 가변 | Transformer 시대에 살아남은 순수 CNN |

### 4.1 ResNet의 한 가지 트릭: skip connection

ResNet 이전에는 30층만 넘어도 학습이 잘 안 됐습니다. 깊을수록 그래디언트가 사라지거나 폭발했죠.

ResNet은 어이없을 정도로 단순한 해결책을 냅니다.

```
입력 x ─┬─→ Conv → ReLU → Conv ─→ (+) → 출력
        │                          ↑
        └──────────────────────────┘   (입력을 그대로 더해줌)
```

수식으로는 `출력 = F(x) + x`. 학습이 어려우면 `F(x)`가 0에 가까워져서 항등함수가 되고, 적어도 입력을 망치진 않습니다. 이 단순한 구조 덕분에 152층, 심지어 1000층까지 안정적으로 학습됩니다.

PyTorch로 잔차 블록을 만들어보면 이 정도입니다.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                   # ← 핵심: 입력을 따로 보관
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity           # ← 마지막에 더하기
        return torch.relu(out)
```

---

## 5. 전이학습(Transfer Learning) — 거인의 어깨에 올라타기

처음부터 ResNet을 학습하려면 ImageNet 백만 장 + GPU 며칠이 필요합니다. 우리는 그렇게 못합니다.

다행히 누군가 이미 학습해둔 모델을 그대로 가져다 쓸 수 있습니다. 이게 **전이학습**입니다.

### 5.1 두 가지 방법

**방법 A: Feature Extractor (특성 추출기)**

사전학습된 모델의 합성곱 부분은 그대로 두고, 마지막 분류기만 새 데이터셋용으로 교체합니다.

```python
import torchvision.models as models

# 사전학습된 ResNet-18 불러오기
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 합성곱 부분 동결 — 학습 안 함
for param in model.parameters():
    param.requires_grad = False

# 분류기만 우리 데이터용으로 교체 (예: 5개 클래스)
model.fc = nn.Linear(model.fc.in_features, 5)

# 이제 model.fc만 학습됨
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

데이터가 수백~수천 장 정도로 적을 때 적합합니다. 학습이 빠르고 과적합이 적습니다.

**방법 B: Fine-tuning (미세 조정)**

모든 층을 학습 가능하게 두되, 학습률을 아주 작게 잡습니다. 사전학습된 가중치를 살짝만 우리 데이터에 맞게 조정한다는 느낌입니다.

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)

# 모든 층 학습, 단 학습률을 매우 낮게
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

데이터가 충분하고(수만 장+) 도메인이 ImageNet과 좀 다를 때 적합합니다.

### 5.2 어느 쪽을 골라야 할까

| 데이터 양 | ImageNet과 도메인 유사도 | 추천 |
|-----------|--------------------------|------|
| 적음 (< 1000장) | 비슷함 (자연 이미지) | Feature Extractor |
| 적음 (< 1000장) | 다름 (의료 영상, 위성 사진) | Feature Extractor + 데이터 증강 |
| 충분 (> 10000장) | 비슷함 | Fine-tuning, lr=1e-5 |
| 충분 (> 10000장) | 다름 | Fine-tuning, lr=1e-4 또는 처음부터 학습 |

---

## 6. 데이터 증강(Data Augmentation) — 한 장을 백 장처럼

이미지 분류는 데이터가 부족할 때가 많습니다. 데이터 증강은 가진 이미지를 살짝 변형해 학습에 활용합니다.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 무작위 크롭 후 리사이즈
    transforms.RandomHorizontalFlip(),       # 좌우 반전
    transforms.ColorJitter(brightness=0.2,   # 색상 흔들기
                          contrast=0.2),
    transforms.RandomRotation(15),           # ±15도 회전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),  # ImageNet 통계
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

**주의할 점**: 검증/테스트에는 증강을 적용하지 않습니다. 평가는 항상 원본에 가까운 형태로 해야 결과가 일관됩니다.

또 도메인에 맞는 증강을 골라야 합니다. 손글씨 숫자(MNIST)에 좌우 반전을 적용하면 "2"가 거꾸로 된 모양이 되어 다른 글자가 됩니다. 항상 "이 변형이 라벨을 바꾸지 않는가?"를 체크하세요.

---

## 7. 실전 예제 — 꽃 5종 분류 (전이학습)

flowers 데이터셋(데이지, 민들레, 장미, 해바라기, 튤립) 5종 분류를 ResNet-18로 풀어봅니다. 데이터가 4000장 정도라 처음부터 학습하기엔 부족합니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --- 1. 데이터 준비 ---
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 폴더 구조: data/train/daisy, data/train/rose, ... 형식이라고 가정
train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds = datasets.ImageFolder("data/val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

# --- 2. 모델 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 합성곱 부분 동결
for p in model.parameters():
    p.requires_grad = False

# 분류기 교체
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

# --- 3. 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(10):
    # 학습
    model.train()
    train_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    # 검증
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += x.size(0)

    print(f"Epoch {epoch+1:2d} | "
          f"train loss {train_loss/total:.4f} acc {correct/total:.4f} | "
          f"val acc {val_correct/val_total:.4f}")
```

이 정도 코드로 보통 검증 정확도 90%를 넘습니다. 처음부터 학습했다면 몇 시간 걸릴 일을, 전이학습으로는 노트북 GPU 10분 안에 끝낼 수 있습니다.

---

## 8. CNN을 쓸 때 흔한 함정

### 8.1 "왜 우리 회사 데이터에선 ImageNet 모델이 잘 안 되지?"

ImageNet은 일상 사진(개, 고양이, 자동차) 위주입니다. 의료 X-ray, 위성 사진, 산업용 결함 검출처럼 도메인이 매우 다른 경우, 사전학습 가중치의 효과가 떨어집니다. 이럴 땐 fine-tuning을 더 깊게 하거나, 도메인용 사전학습 모델(예: 의료영상이면 RadImageNet)을 찾아보세요.

### 8.2 입력 정규화 안 함

사전학습 모델은 ImageNet 평균/표준편차로 정규화된 입력에 맞춰 학습됐습니다. `transforms.Normalize`를 빼먹으면 정확도가 30%포인트씩 깎입니다.

### 8.3 흑백 이미지를 모델에 그냥 넣음

ResNet은 3채널 입력을 기대합니다. 흑백 이미지는 같은 채널을 3번 복사하든가, 첫 합성곱 층을 1채널 입력용으로 교체해야 합니다.

```python
# 흑백 1채널 이미지를 3채널처럼 만들기
gray_to_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
```

### 8.4 학습률 한 가지로 다 처리

전이학습할 때 합성곱 부분과 새로 만든 분류기에 같은 학습률을 쓰면 안 됩니다. 분류기는 1e-3, 합성곱은 1e-5 식으로 차등을 두는 게 안전합니다.

```python
optimizer = optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-3},
    {"params": [p for n, p in model.named_parameters() if "fc" not in n],
     "lr": 1e-5},
])
```

---

## 9. 한눈에 정리

| 개념 | 한 줄 요약 |
|------|-----------|
| 합성곱 | 작은 필터를 슬라이딩하며 패턴 찾기 |
| stride | 필터 이동 간격 (보통 1, 줄이려면 2) |
| padding | 가장자리 0 덧대기 (보통 `(kernel-1)/2`) |
| Max Pool | 영역 내 최댓값으로 압축, 보통 `2×2` |
| BatchNorm | 학습 안정화 (합성곱 다음에 거의 항상) |
| ResNet skip | `출력 = F(x) + x`, 깊은 망 학습 가능 |
| 전이학습 | 사전학습 가중치 활용, 데이터 적을 때 필수 |
| 데이터 증강 | 한 장으로 여러 변형 만들어 학습 |

---

## 더 깊이 가고 싶으시면

- **Object Detection**: 이미지에서 "어디에" "무엇이" 있는지. YOLO, Faster R-CNN, DETR
- **Segmentation**: 픽셀 단위로 분류. U-Net, Mask R-CNN, SAM(Segment Anything)
- **Vision Transformer (ViT)**: CNN 없이 Transformer만으로 이미지 분류. 큰 데이터셋에서 CNN을 능가
- **Self-Supervised Learning**: 라벨 없이도 좋은 특성 학습. SimCLR, MoCo, DINO
- **확산 모델 (Diffusion)**: 이미지 생성. Stable Diffusion, DALL·E 시리즈

다음 장에서는 시계열과 텍스트를 다루는 RNN/LSTM/GRU를 살펴봅니다. CNN이 "공간"의 패턴을 잡는다면, RNN은 "시간"의 패턴을 잡는 도구입니다.
