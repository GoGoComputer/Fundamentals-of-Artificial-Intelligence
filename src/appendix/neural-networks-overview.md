# 부록: 신경망 가족 한눈에 — MLP, CNN, RNN, Transformer

> 우리는 4~5장에서 **MLP** (가장 기본 신경망)만 다뤘어요.
> 하지만 신경망에는 다양한 종류가 있어요.
> 각각이 어떤 데이터에 어떤 일을 잘하는지 한눈에 정리합니다.
>
> 깊이 있는 내용은 아니에요. **다음에 무엇을 배울지** 가이드입니다.

---

## 큰 가족도

```
┌─────────────────────────────────────────────────┐
│             신경망 (Neural Networks)               │
│                                                   │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐ │
│  │  MLP   │  │  CNN   │  │  RNN   │  │  자기   │ │
│  │ 표 데이터│  │ 이미지  │  │ 시퀀스  │  │ 어텐션  │ │
│  └────────┘  └────────┘  └────────┘  └────────┘ │
│                                          │        │
│                                          ↓        │
│                                  ┌──────────────┐ │
│                                  │ Transformer  │ │
│                                  │ 거의 모든 것  │ │
│                                  └──────────────┘ │
└─────────────────────────────────────────────────┘
```

각각 차례로 봅시다.

---

## 1. MLP (Multi-Layer Perceptron)

우리가 4~5장에서 다룬 거예요. 가장 기본.

### 무엇?

여러 개의 Dense (Fully Connected) 층을 쌓은 신경망.

```
입력 ─→ [Dense] ─→ [Dense] ─→ [Dense] ─→ 출력
```

### 적합한 데이터

- **표 형태 데이터** (엑셀 같은)
- 작은~중간 크기

### PyTorch

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

### 한계

- 이미지의 공간 구조 못 잡음 (픽셀을 1D로 펴야 함)
- 시퀀스의 순서 못 잡음
- 큰 입력에 파라미터 폭발

---

## 2. CNN (Convolutional Neural Network)

**이미지 처리의 표준.**

### 핵심 아이디어

> "이미지의 한 부분만 보면 전체를 못 알아본다. **작은 영역을 슬라이드**하면서 특징을 추출하자."

```
이미지:           Convolution 필터:    출력 (특징 맵):
[ ][ ][ ][ ]      [1][0][1]              [ ][ ]
[ ][ ][ ][ ]   ⊗  [0][1][0]   →           [ ][ ]
[ ][ ][ ][ ]      [1][0][1]
[ ][ ][ ][ ]
```

필터를 슬라이드하면서 특징(가장자리, 모양 등)을 잡아냄.

### 핵심 층

| 층 | 역할 |
|---|------|
| **Conv2d** | 컨볼루션 (특징 추출) |
| **MaxPool2d** | 다운샘플링 (크기 줄임) |
| **BatchNorm2d** | 정규화 |
| **Linear** | 마지막 분류 층 |

### PyTorch 예시

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        return self.fc(x)
```

### 유명한 CNN들

| 모델 | 연도 | 핵심 기여 |
|------|------|----------|
| LeNet | 1998 | 최초의 실용 CNN |
| AlexNet | 2012 | ImageNet 우승, 딥러닝 부흥 |
| VGG | 2014 | 단순한 깊은 구조 |
| **ResNet** | 2015 | Skip Connection (깊이 100층+) |
| EfficientNet | 2019 | 효율적인 스케일링 |
| ConvNeXt | 2022 | 현대적 개선 |

### 적합한 데이터

- **이미지** (분류, 탐지, 분할)
- 1D Convolution은 시계열에도

### 다음 학습

- **PyTorch torchvision**: ResNet, EfficientNet 등 사전학습 모델
- 책: "Deep Learning with PyTorch"
- 강의: Stanford CS231n

---

## 3. RNN / LSTM / GRU

**순차적 데이터 (시계열, 텍스트) 처리.**

### 핵심 아이디어

> "이전 단계의 출력을 다음 단계의 입력으로." (순환 구조)

```
시간:     t=1     t=2     t=3     t=4
            ↓       ↓       ↓       ↓
입력:     x1      x2      x3      x4
            ↓       ↓       ↓       ↓
RNN:    [hidden] [hidden] [hidden] [hidden]
           ↓ ─────→ ↓ ─────→ ↓ ─────→ ↓
출력:     y1      y2      y3      y4
```

`hidden`이 "기억"을 다음 단계로 전달.

### 한계

기본 RNN은 **긴 시퀀스에서 옛날 정보를 잘 기억 못 해요** (Vanishing Gradient).

→ **LSTM (Long Short-Term Memory)** 와 **GRU (Gated Recurrent Unit)** 가 이를 개선.

### PyTorch

```python
# 가장 단순
self.rnn = nn.RNN(input_size=10, hidden_size=64, num_layers=2)

# LSTM (더 자주)
self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)

# GRU (LSTM보다 가벼움)
self.gru = nn.GRU(input_size=10, hidden_size=64, num_layers=2)
```

### 적합한 데이터

- 시계열 (주가, 매출, 날씨)
- 텍스트 (옛날 NLP 표준)
- 음성

### 현대의 위치

**Transformer가 거의 다 대체**했어요. 다만:

- **시계열 예측**: LSTM 여전히 자주 사용
- **모바일 / 임베디드**: 가벼워서 사용
- **단순한 시퀀스 작업**: 충분

---

## 4. Transformer

**현대 AI의 거의 전부.** ChatGPT, BERT, ViT, Stable Diffusion — 다 Transformer 기반.

### 핵심 아이디어: Self-Attention

> "시퀀스의 각 단어가 다른 모든 단어와 얼마나 관련 있는지 직접 계산."

RNN은 순차적으로 정보를 전달하지만, Transformer는 **모든 위치를 한꺼번에** 봅니다.

```
"고양이가 매트 위에 앉아 있다"
  ↓     ↓    ↓    ↓    ↓
[모든 단어가 모든 단어와 attention 계산]
  ↓     ↓    ↓    ↓    ↓
[정보가 풍부해진 새로운 표현]
```

### 왜 강력해요?

1. **병렬 처리 가능** (RNN은 순차적)
2. **긴 의존성** 잘 잡음
3. **확장성**: 데이터/모델 키울수록 좋아짐

### 핵심 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| **Self-Attention** | 시퀀스 내 관계 계산 |
| **Multi-Head Attention** | 여러 종류의 관계 동시 |
| **Positional Encoding** | 순서 정보 주입 |
| **Feed-Forward** | 비선형 변환 |
| **LayerNorm** | 정규화 |

### Transformer 가족

```
Transformer (2017)
├─ Encoder만 사용
│  ├─ BERT (2018) — 텍스트 이해
│  └─ ViT (2020) — 이미지 분류 (Vision Transformer)
│
├─ Decoder만 사용
│  ├─ GPT-2 (2019)
│  ├─ GPT-3 (2020)
│  ├─ GPT-4 (2023)
│  └─ Llama, Claude, Gemini, ...
│
└─ Encoder + Decoder
   ├─ T5 (2019) — 텍스트 변환
   └─ BART (2019) — 요약 등
```

### 코드 한 줄

직접 만드는 건 어려워요. **Hugging Face가 표준.**

```python
from transformers import AutoModel, AutoTokenizer

# 사전학습 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# 사용
inputs = tokenizer("Hello world", return_tensors='pt')
outputs = model(**inputs)
```

이 한 패턴으로 BERT, RoBERTa, GPT, T5 등 모두 사용 가능.

### 적합한 데이터

- **모든 시퀀스 데이터** (텍스트, 시계열, 코드, 음악)
- **이미지** (ViT, Swin)
- **멀티모달** (이미지 + 텍스트)

### 다음 학습

- **Hugging Face Course** (무료, 한국어도 있음)
- 책: "Natural Language Processing with Transformers"
- 강의: Stanford CS224n

---

## 5. 비교표

| | MLP | CNN | RNN/LSTM | Transformer |
|---|---|---|---|---|
| **잘하는 데이터** | 표 | 이미지 | 시퀀스 | 모든 것 |
| **공간 구조 인식** | ❌ | ✅ | △ | ✅ |
| **시간 순서 인식** | ❌ | △ (1D) | ✅ | ✅ |
| **장거리 의존성** | ❌ | △ | △ | ✅✅ |
| **병렬 학습** | ✅ | ✅ | ❌ | ✅✅ |
| **데이터 요구량** | 작음 | 중간 | 중간 | **매우 큼** |
| **학습 비용** | 낮음 | 보통 | 보통 | **매우 큼** |
| **현대 사용** | 표 데이터 | 이미지 | 일부 시계열 | **모든 것** |

---

## 6. "그럼 우리는 무엇을 배워야 해요?"

목적과 시간에 따라 다르지만, 실용적 순서:

### 트랙 A: 일반 ML 엔지니어

```
MLP (이 자료에서 끝남)
  ↓
CNN (이미지 다루면)
  ↓
Transformer (현대 표준)
  ↓
Pre-trained Models 활용 (Hugging Face)
```

### 트랙 B: NLP 엔지니어

```
MLP → Transformer (직진)
  ↓
BERT 활용
  ↓
GPT API 활용
  ↓
LLM Fine-tuning
```

### 트랙 C: CV 엔지니어

```
MLP → CNN
  ↓
ResNet, EfficientNet
  ↓
ViT (Vision Transformer)
  ↓
Diffusion Models (이미지 생성)
```

### 트랙 D: 시계열 엔지니어

```
MLP
  ↓
LSTM
  ↓
Temporal Fusion Transformer
  ↓
Foundation Models (TimesFM 등)
```

---

## 7. 추천 라이브러리

### 사전학습 모델 가져다 쓰기

- **Hugging Face Transformers** ⭐⭐⭐ (텍스트/이미지/오디오)
- **torchvision** (이미지)
- **timm** (이미지, 사전학습 모델 모음)

### 처음부터 만들기

- PyTorch
- PyTorch Lightning (보일러플레이트 줄임)

### LLM 응용

- LangChain, LlamaIndex (RAG, Agents)
- vLLM (빠른 추론)

---

## 8. "직접 만들어 봐야 해요?"

**처음에는 사전학습 모델을 가져다 쓰세요.** 직접 만드는 건 매우 비싸요.

### Pretrained Model 활용 — 권장

```python
# Image classification
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# NLP
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```

이걸 본인 데이터로 **Fine-tuning**하시는 게 표준.

### From Scratch — 학습용

직접 만드는 건 학습 목적으로만. ResNet from scratch, GPT from scratch 같은 튜토리얼이 많음.

---

## 정리

```
신경망의 가족:

1. MLP — 표 데이터 (이 자료)
2. CNN — 이미지
3. RNN/LSTM — 시퀀스 (구식)
4. Transformer — 거의 모든 것 (현대)

다음 단계:
- 사전학습 모델 활용 먼저
- Hugging Face가 사실상 표준
- 본인 분야에 맞춰 선택
```

이 자료의 MLP를 잘 익히셨으면 다른 신경망도 같은 패턴으로 만드실 수 있어요. 가장 어려운 건 **첫 번째 신경망을 직접 학습시켜 본 경험** 자체예요.

➡️ [더 나아가기](더-나아가기.md)
➡️ [메인으로](../README.md)
