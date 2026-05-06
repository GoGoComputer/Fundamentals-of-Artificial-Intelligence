# 5.4 무기 2: Dropout

## "그냥 끄세요" 아이디어

Dropout은 2012년에 제프 힌턴(Geoffrey Hinton)이 제안한, **단순하지만 천재적인 아이디어**예요.

> **학습 중에 일부 뉴런을 무작위로 꺼라.**

끝입니다. 진짜로요. 이게 어떻게 도움이 되는지가 흥미로운 부분이에요.

---

## 왜 효과가 있어요?

### 비유 1: 한 명에게만 의존하지 않기

축구팀이 매번 같은 11명으로 경기한다고 해 봐요. 그 중 한 명이 다치면? 팀 전체가 무너져요.

대신 매 경기마다 **무작위로 선수를 빼는 훈련**을 하면 어떨까요? 모든 선수가 본인의 자리뿐 아니라 다른 자리도 할 줄 알게 됩니다. **팀이 더 강건해져요.**

신경망도 똑같아요. Dropout은 모델이 **특정 뉴런 몇 개에만 의존하지 못하게** 강제해요. 모든 뉴런이 독립적으로 일을 할 줄 알게 됩니다.

### 비유 2: 여러 모델의 앙상블

매 학습 step마다 다른 부분이 꺼져요. 사실상 **매번 다른 모델을 학습하는 것**과 같아요. 결과적으로 여러 모델의 평균을 내는 효과가 생깁니다.

앙상블이 단일 모델보다 잘 한다는 건 ML의 정설이에요. 랜덤 포레스트가 단일 트리보다 잘 한 것처럼.

---

## 그림으로

원래 신경망 (Dropout 없음):
```
입력 ─→ ●─●─●─●─● ─→ ●─●─●─●─● ─→ 출력
        모든 연결 살아 있음
```

Dropout 있음 (학습 중):
```
입력 ─→ ●─×─●─●─× ─→ ×─●─●─×─● ─→ 출력
        무작위로 일부 끔 (× = 꺼짐)
        매 step마다 다른 부분이 꺼짐
```

**중요: 평가 시에는 모두 켜져 있어요.** Dropout은 학습할 때만 동작합니다.

---

## 코드: 한 줄만 추가

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.dropout1 = nn.Dropout(0.3)    # 30% 끄기
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)            # ← 여기서 끔
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x).squeeze()
```

`nn.Dropout(0.3)` 한 줄이 끝이에요. 0.3은 **각 뉴런이 꺼질 확률**입니다.

### Sequential로 더 짧게

```python
model = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(32, 1),
)
```

---

## 어디에 둬야 해요?

**활성화 함수(ReLU) 다음**에 두는 게 표준이에요.

```python
# 표준 패턴
x = self.fc1(x)         # Linear
x = torch.relu(x)        # 활성화
x = self.dropout(x)      # Dropout!
```

**마지막 층 다음에는 두지 마세요.**

```python
# ❌ 잘못
self.fc3 = nn.Linear(32, 1)
self.dropout3 = nn.Dropout(0.3)    # 출력에 dropout?
```

---

## p (drop 비율)를 얼마로?

`p`는 각 뉴런이 꺼질 확률이에요.

| p 값 | 효과 |
|-----|------|
| 0.1 | 약한 정규화 |
| 0.2-0.3 | **표준** (대부분 여기) |
| 0.5 | 강한 정규화 (Hinton 원논문 권장) |
| 0.7+ | 너무 강함 (학습 안 됨 위험) |

### 실용 가이드

- **첫 시도: 0.3**
- 과적합이 심하면 → 0.5
- 학습이 안 되면 (loss 안 떨어짐) → 0.1

---

## 학습 vs 평가에서의 차이

⚠️ **이게 정말 중요합니다.** Dropout은 모드에 따라 다르게 동작해요.

### 학습 모드 (`model.train()`)
- Dropout 활성화: 일부 뉴런이 무작위로 꺼짐

### 평가 모드 (`model.eval()`)
- Dropout 비활성화: 모든 뉴런이 켜짐

만약 평가할 때 `model.eval()`을 안 하면, 매번 다른 결과가 나와요!

```python
# ❌ 잘못 (Dropout 켜진 채 평가)
model = MyModel()
model.train()    # 또는 모드 안 바꿈
output1 = model(test_x)
output2 = model(test_x)
print(output1 == output2)    # False! 매번 다름

# ✅ 올바름
model.eval()
output = model(test_x)    # 항상 같은 결과
```

---

## 성능 비교 실험

같은 데이터, 같은 모델 구조로 Dropout만 다르게 시도해 봅시다.

```python
# 실험 1: Dropout 없음
class NoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()


# 실험 2: Dropout 0.3
class WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x).squeeze()


# 둘 다 같은 학습 코드로 학습 후 비교
```

전형적인 결과:

| 모델 | Train Loss | Val Loss | Gap |
|------|-----------|---------|-----|
| Dropout 없음 | 0.05 | 0.45 | **0.40 (과적합)** |
| Dropout 0.3 | 0.15 | 0.30 | **0.15 (개선)** |

Train Loss는 약간 올라갔지만(학습이 더 어려워졌으니), **Val Loss가 더 좋아졌어요.** 이게 정규화의 효과예요.

---

## Dropout 변형들

### Dropout1d, Dropout2d, Dropout3d

CNN에서 채널 단위로 끄는 변형들이 있어요.

```python
nn.Dropout()      # 1D 입력 (Linear 다음)
nn.Dropout1d()    # 1D 시퀀스
nn.Dropout2d()    # 이미지 채널 (CNN)
```

지금은 그냥 `nn.Dropout()`만 알아도 됩니다.

### AlphaDropout

SELU 활성화 함수와 같이 쓰는 특별한 Dropout. 거의 안 써요.

---

## Dropout vs Batch Normalization

둘 다 정규화 기법이지만 다른 일을 해요.

| | Dropout | BatchNorm |
|---|---|---|
| 무엇? | 뉴런을 무작위로 끔 | 배치 단위로 정규화 |
| 효과 | 과적합 방지 | 학습 안정화 + 빠른 수렴 |
| 위치 | 보통 활성화 후 | 보통 활성화 전 |
| eval 모드 | 비활성화 | 학습 통계 사용 |

**둘 다 같이 쓸 수 있어요.** 다만 순서에 주의:

```python
# 권장 순서
Linear → BatchNorm → ReLU → Dropout
```

---

## 함정과 주의사항

### 함정 1: 작은 모델에는 효과 적음

매개변수가 적은 모델에는 Dropout 효과가 적어요. 어차피 과적합 위험이 적으니까요.

### 함정 2: 너무 많이 쓰면 안 됨

모든 층에 0.5씩 적용하면 학습이 거의 안 돼요. 보통 한두 곳에만 적용.

### 함정 3: 마지막 층 직전 X

```python
# ❌
self.fc_last = nn.Linear(32, 1)
self.dropout_before_last = nn.Dropout(0.3)
```

마지막 출력 직전에는 Dropout을 두지 마세요. 의미가 없거나 해롭습니다.

### 함정 4: 평가 시 .eval() 안 함

```python
# ❌ 잘못
output = model(x)
metrics = compute(output)    # Dropout 켜져 있어서 매번 다름

# ✅ 올바름
model.eval()
with torch.no_grad():
    output = model(x)
```

---

## 정리

```python
# 모델 정의에 한 줄 추가
nn.Dropout(0.3)    # 30% 확률로 끄기

# 위치: 활성화 함수 다음
x = torch.relu(layer(x))
x = self.dropout(x)

# 평가 시
model.eval()    # Dropout 자동으로 끔
```

**한 줄 요약: "각 층 사이에 Dropout(0.3)을 두고, 평가 전엔 model.eval()"**

---

## 자주 묻는 질문

> **Q. 모든 층 사이에 다 둬야 해요?**
>
> 아니에요. 보통 hidden layer 사이에만. 마지막 층 직전에는 X.

> **Q. 학습이 너무 흔들려요. Dropout 때문일까요?**
>
> 가능해요. p를 줄여보세요 (0.3 → 0.1). 또는 학습률을 줄이세요.

> **Q. inverted dropout이 뭐예요?**
>
> 학습 시 켜진 뉴런의 출력을 1/(1-p) 배 키우는 기법이에요. PyTorch는 자동으로 이걸 해요. 우리가 신경 쓸 필요 없습니다.

➡️ 다음: [5.5 무기 3: Weight Decay (L2)](05-Weight-Decay.md)
