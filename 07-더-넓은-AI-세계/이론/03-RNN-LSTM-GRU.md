# 7.3 RNN · LSTM · GRU — 시간을 기억하는 신경망

## 들어가며

CNN이 "공간"의 패턴을 잡는 도구였다면, RNN은 "시간"의 패턴을 잡는 도구입니다. 시간에 따라 흐르는 데이터는 생각보다 우리 주변에 많습니다.

- 주가 차트: 어제까지의 흐름을 보고 내일을 예측
- 음성: 1초 전의 발음이 지금 발음의 의미를 결정
- 문장: "나는 사과를 ___" 빈칸 앞 단어들이 다음 단어를 결정
- 센서 로그: 진동의 시간 패턴으로 기계 고장 예측

이런 데이터의 공통점은 **순서가 의미를 만든다**는 것입니다. "오늘 비가 와서 우산을 챙겼다"와 "우산을 챙겼더니 오늘 비가 왔다"는 단어는 같지만 의미가 다르죠.

MLP나 CNN은 순서 정보를 거의 활용하지 못합니다. RNN은 이 빈자리를 메우려고 만들어졌습니다.

> 🧮 **수학 보충** — 이번 장의 핵심은 **재귀(recursion)**입니다. 이전 출력을 다음 입력에 다시 넣는 구조죠. [부록/수학/04-미적분.md](../../부록/수학/04-미적분.md)의 합성함수 미분(연쇄 법칙)을 이해하면 학습 과정이 훨씬 또렷해집니다.

---

## 1. RNN 기본 구조 — 메모를 가진 신경망

### 1.1 직관: 책을 읽으면서 메모하기

긴 소설을 읽을 때 우리는 매 페이지마다 책 전체를 다시 읽지 않습니다. 머릿속에 "지금까지의 줄거리"를 요약해두고, 새 페이지를 그 요약과 함께 이해합니다.

RNN의 작동 방식이 똑같습니다.

```
시점 t의 처리:
  새 입력 x_t  +  이전까지의 메모 h_{t-1}  →  새 메모 h_t  →  출력 y_t
```

수식으로는

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

복잡해 보이지만 의미는 단순합니다. "이번 입력"과 "지금까지 기억"을 섞어서 새 기억을 만든다는 것이 전부입니다.

### 1.2 펼쳐서 보기

같은 RNN 셀이 시간에 따라 반복됩니다.

```
x_1 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3 → ...
              ↑ x_2 ↑       ↑ x_3 ↑       ↑ x_4 ↑
```

여기서 중요한 점: **모든 시점에서 같은 가중치 W_x, W_h를 씁니다.** CNN의 필터 공유와 비슷한 발상입니다. "어떤 시점이든 입력을 처리하는 방식은 동일하다"는 가정이죠.

### 1.3 PyTorch로 가볍게 써보기

```python
import torch
import torch.nn as nn

# 입력 차원 10, 은닉 차원 20
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# (배치 크기, 시퀀스 길이, 입력 차원)
x = torch.randn(2, 5, 10)   # 5단계 시퀀스
out, h = rnn(x)

print(out.shape)   # torch.Size([2, 5, 20])  — 매 시점의 출력
print(h.shape)     # torch.Size([1, 2, 20])  — 마지막 시점의 hidden state
```

`out`은 매 시점의 출력 전부, `h`는 마지막 시점의 메모(hidden state)입니다. 분류 문제라면 보통 `h`만 쓰고, 시계열을 시계열로 변환하는 문제라면 `out`을 씁니다.

---

## 2. RNN의 치명적 한계 — 장기 의존성 문제

### 2.1 문제 상황

긴 문장을 생각해보세요.

> "나는 어렸을 때 프랑스에서 5년을 살았는데, 그래서 지금도 가끔 ___로 꿈을 꾼다."

빈칸에 들어갈 말은 "프랑스어"입니다. 그런데 이 단서("프랑스에서")는 빈칸과 20단어쯤 떨어져 있습니다. 기본 RNN은 이렇게 멀리 떨어진 정보를 기억하기 어렵습니다.

### 2.2 왜 이런 일이 벌어지나

RNN을 학습할 때 그래디언트가 시간을 거꾸로 거슬러 올라갑니다(BPTT, Backpropagation Through Time). 이때 매 시점마다 같은 가중치 행렬이 곱해지는데, 이 값이 1보다 작으면 시간을 거슬러 올라갈수록 작아져서 결국 0에 가까워집니다.

```
20단계 거슬러 올라가면:
  0.9^20 ≈ 0.12   (그래디언트 사라짐)
  1.1^20 ≈ 6.7    (그래디언트 폭발)
```

이걸 **그래디언트 소실/폭발(vanishing/exploding gradient)** 문제라고 부릅니다. 결과적으로 기본 RNN은 **5~10 시점 이상의 의존성을 잘 학습하지 못합니다.**

---

## 3. LSTM — 게이트로 기억 관리하기

### 3.1 핵심 아이디어

1997년 Hochreiter와 Schmidhuber가 제안한 LSTM(Long Short-Term Memory)은 RNN에 **게이트(gate)**를 추가합니다. 게이트는 0~1 사이 값을 출력하는 작은 신경망인데, "이 정보를 얼마나 통과시킬까?"를 결정합니다.

LSTM에는 세 종류의 게이트가 있습니다.

| 게이트 | 역할 | 비유 |
|--------|------|------|
| Forget gate (`f_t`) | 오래된 기억 중 무엇을 잊을지 | 책장 정리 |
| Input gate (`i_t`) | 새 정보 중 무엇을 저장할지 | 메모지 분류 |
| Output gate (`o_t`) | 저장된 기억 중 무엇을 출력할지 | 대답할 내용 고르기 |

### 3.2 셀 상태(cell state) — 장기 기억의 통로

LSTM의 진짜 비밀은 hidden state 외에 **cell state**라는 또 다른 통로가 있다는 점입니다.

```
cell state c_t:  과거 정보가 거의 손실 없이 흘러가는 고속도로
hidden state h_t: 매 시점에 노출되는 단기 기억
```

cell state는 게이트의 곱셈만 거치고 활성화 함수를 거치지 않아서, 그래디언트가 소실되지 않고 멀리까지 전달됩니다. 이게 LSTM이 장기 의존성을 학습할 수 있는 이유입니다.

### 3.3 PyTorch로 보면

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(2, 5, 10)
out, (h, c) = lstm(x)   # LSTM은 hidden과 cell 둘 다 반환

print(out.shape)   # torch.Size([2, 5, 20])
print(h.shape)     # torch.Size([1, 2, 20])
print(c.shape)     # torch.Size([1, 2, 20])
```

사용법은 RNN과 거의 같지만 반환값이 `(h, c)` 튜플입니다.

---

## 4. GRU — LSTM의 단순화 버전

2014년 Cho 등이 제안한 GRU(Gated Recurrent Unit)는 LSTM을 단순화한 모델입니다.

### 4.1 LSTM과 비교

| 항목 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3개 (forget, input, output) | 2개 (reset, update) |
| 별도 cell state | 있음 | 없음 (hidden state로 통합) |
| 파라미터 수 | 많음 | LSTM의 약 75% |
| 학습 속도 | 느림 | 빠름 |
| 성능 | 데이터 많을 때 약간 우세 | 데이터 적을 때 약간 우세 |

실무에서는 둘 다 시도해보고 검증 성능이 좋은 쪽을 택하는 게 정답입니다. 차이가 크지 않은 경우가 많습니다.

### 4.2 PyTorch에서

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(2, 5, 10)
out, h = gru(x)   # cell state 없음, RNN과 같은 인터페이스
```

---

## 5. 양방향 RNN — 미래도 보기

지금까지의 RNN은 과거 → 현재 방향으로만 정보가 흐릅니다. 그런데 어떤 문제는 미래도 봐야 합니다.

> "나는 ___를 먹었다. 사과는 빨갛고 달았다."

빈칸에 "사과"가 들어간다는 걸 알려면 다음 문장까지 봐야 하죠. 이럴 때 **양방향 RNN(Bidirectional RNN)**을 씁니다.

```python
bilstm = nn.LSTM(input_size=10, hidden_size=20,
                 bidirectional=True, batch_first=True)

x = torch.randn(2, 5, 10)
out, (h, c) = bilstm(x)

print(out.shape)   # torch.Size([2, 5, 40])  — 정방향 20 + 역방향 20
```

출력 차원이 두 배로 늘어납니다. 분류, 개체명 인식, 품사 태깅처럼 시퀀스 전체가 이미 주어진 상황에서는 양방향이 거의 항상 더 좋습니다. 단, 실시간 예측처럼 미래를 못 보는 상황에는 쓸 수 없습니다.

---

## 6. 시퀀스를 다루는 세 가지 패턴

### 6.1 시퀀스 → 단일값 (many-to-one)

문장 감성 분류, 시계열 다음값 예측 같은 문제입니다.

```
"이 영화 정말 재밌었다"  →  [긍정/부정]
```

마지막 hidden state만 분류기에 통과시킵니다.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # (B, L) → (B, L, E)
        _, (h, _) = self.lstm(x)           # h: (1, B, H)
        h = h.squeeze(0)                   # (B, H)
        return self.fc(h)
```

### 6.2 시퀀스 → 시퀀스 (many-to-many, 같은 길이)

품사 태깅, 개체명 인식처럼 입력 단어마다 출력을 내야 하는 문제입니다.

```
"나는  학교에  간다"
 ↓     ↓       ↓
 [N]   [N]    [V]
```

매 시점의 출력을 분류기에 통과시킵니다.

```python
class POSTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)  # 양방향이라 *2

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)              # (B, L, 2H)
        return self.fc(out)                # (B, L, num_tags)
```

### 6.3 시퀀스 → 시퀀스 (다른 길이) — Encoder-Decoder

번역, 요약, 챗봇처럼 입출력 길이가 다른 문제입니다.

```
입력 인코더:  "나는 학교에 간다"  →  [요약된 의미 벡터]
                                    ↓
출력 디코더:  [의미 벡터]  →  "I  go  to  school"
```

이 구조가 발전해서 결국 Transformer로 이어집니다. 다음 장에서 자세히 다룹니다.

---

## 7. 실전 예제 — 영화 리뷰 감성 분류

IMDB 영화 리뷰 데이터셋(긍정/부정 50000건)을 LSTM으로 분류해봅니다.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# --- 1. 토크나이저와 단어 사전 ---
tokenizer = get_tokenizer("basic_english")
train_iter = IMDB(split="train")

def yield_tokens(iter):
    for label, text in iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=["<unk>", "<pad>"],
    max_tokens=20000
)
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab["<pad>"]

# --- 2. 데이터 변환 함수 ---
def text_pipeline(text):
    return torch.tensor([vocab[token] for token in tokenizer(text)[:200]])

def label_pipeline(label):
    return 1 if label == "pos" else 0

def collate_fn(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        texts.append(text_pipeline(text))
    labels = torch.tensor(labels)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts, labels

# --- 3. 모델 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           num_layers=2, dropout=0.3,
                           bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        # h: (num_layers*2, B, H) — 마지막 층의 양방향만 사용
        h = torch.cat([h[-2], h[-1]], dim=1)   # (B, 2H)
        h = self.dropout(h)
        return self.fc(h)

# --- 4. 학습 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier(len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(IMDB(split="train"), batch_size=64,
                         shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(IMDB(split="test"), batch_size=64,
                        collate_fn=collate_fn)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← 중요
        optimizer.step()

    # 평가
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1} | test acc {correct/total:.4f}")
```

이 정도면 보통 87~89% 정확도가 나옵니다. 요즘 BERT 기반 모델(95%+)에 비하면 낮지만, 학습 시간과 자원을 고려하면 합리적입니다.

### 7.1 코드의 핵심 포인트

1. **`padding_idx` 명시** — 길이가 다른 문장을 같은 배치에 넣으려고 `<pad>`로 채울 때, 이 토큰이 학습에 영향을 주지 않도록 합니다.
2. **`clip_grad_norm_`** — RNN 계열은 그래디언트 폭발이 흔합니다. 그래디언트의 노름을 1.0으로 제한합니다.
3. **양방향 + 다층** — `num_layers=2, bidirectional=True`로 표현력을 높입니다. 그만큼 dropout으로 과적합 방지가 필요합니다.

---

## 8. 시계열 예측 예제 — 일별 온도 다음 날 예측

```python
import numpy as np
import torch
import torch.nn as nn

# 가짜 시계열 만들기 (실제로는 CSV 등에서 로드)
np.random.seed(0)
T = 1000
t = np.arange(T)
data = np.sin(t * 0.05) + 0.5 * np.sin(t * 0.13) + 0.1 * np.random.randn(T)

# 슬라이딩 윈도우로 데이터셋 만들기
WINDOW = 30
X, y = [], []
for i in range(T - WINDOW):
    X.append(data[i:i+WINDOW])
    y.append(data[i+WINDOW])
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, 30, 1)
y = torch.tensor(y, dtype=torch.float32)

# 학습/검증 분할
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 모델
class TimeSeriesGRU(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0)).squeeze(-1)

model = TimeSeriesGRU()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(50):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        print(f"Epoch {epoch+1:2d} | train {loss.item():.4f} | val {val_loss:.4f}")
```

윈도우 크기, 은닉 차원, 학습률을 바꿔가며 검증 손실이 가장 낮은 조합을 찾는 게 시계열 모델링의 본질입니다.

---

## 9. RNN 계열을 쓸 때 알아둘 점

### 9.1 패딩과 마스킹

배치마다 시퀀스 길이가 다르면 짧은 것을 `<pad>`로 채워야 합니다. PyTorch는 `pack_padded_sequence`로 이를 효율적으로 처리할 수 있습니다.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 길이 정보가 있어야 함
lengths = torch.tensor([10, 7, 5])  # 각 배치 원소의 실제 길이

packed = pack_padded_sequence(x, lengths, batch_first=True,
                              enforce_sorted=False)
out, h = lstm(packed)
out, _ = pad_packed_sequence(out, batch_first=True)
```

이걸 안 쓰면 모델이 `<pad>` 토큰까지 처리하느라 시간을 낭비하고, hidden state가 오염됩니다.

### 9.2 Truncated BPTT

문서 전체를 한 번에 RNN에 넣으면 메모리가 폭발합니다. 긴 시퀀스는 일정 길이로 잘라서(예: 200 토큰씩) 학습하되, hidden state를 다음 청크로 이어주는 방식을 씁니다. 언어 모델 학습에서 표준입니다.

### 9.3 RNN의 시대는 끝났는가?

NLP에서는 거의 끝났습니다. 2017년 Transformer 등장 이후 BERT, GPT 등이 RNN을 대체했습니다. 하지만 다음 영역에서는 여전히 유효합니다.

- **짧은 시계열**: 온도, 매출처럼 길이 100 이하 시퀀스
- **온라인 학습**: 실시간으로 한 번에 한 입력씩 처리
- **저자원 환경**: 모바일/엣지 디바이스에서 작은 모델이 필요할 때
- **간단한 베이스라인**: Transformer 도입 전 빠른 비교군

### 9.4 RNN vs Transformer, 한 줄 요약

- RNN: 순차 처리, 메모리 적게 씀, 짧은 시퀀스에 효율적
- Transformer: 병렬 처리, 메모리 많이 씀, 긴 의존성 학습 우수

---

## 10. 한눈에 정리

| 모델 | 특징 | 언제 쓰나 |
|------|------|-----------|
| RNN | 가장 단순, 장기 의존성 약함 | 교육용, 매우 짧은 시퀀스 |
| LSTM | 게이트 3개, cell state | 길이 50~200 시퀀스, 안정적 선택 |
| GRU | 게이트 2개, 단순화 | LSTM 대안, 데이터 적을 때 |
| Bi-LSTM | 양방향 | 시퀀스 분류, 태깅 (실시간 X) |

**필수 체크리스트**
- [ ] 그래디언트 클리핑 (`clip_grad_norm_`) 했는가
- [ ] 패딩 토큰 마스킹 했는가
- [ ] 입력 정규화 (시계열) 또는 임베딩 (텍스트) 했는가
- [ ] 검증 손실로 epoch 정했는가 (RNN은 과적합 빠름)

---

## 더 깊이 가고 싶으시면

- **Seq2Seq + Attention**: RNN 인코더-디코더에 어텐션을 추가. Transformer의 직전 단계
- **Word Embeddings**: Word2Vec, GloVe, FastText — 단어를 벡터로
- **Temporal Convolutional Networks**: CNN으로 시계열 다루기, RNN보다 빠름
- **State Space Models (Mamba 등)**: RNN의 부활. 매우 긴 시퀀스에서 Transformer 대안
- **Time Series Foundation Models**: TimesFM, Chronos 등 시계열 사전학습 모델

다음 장에서는 RNN과 Attention의 자식인 **Transformer**를 살펴봅니다. 현재 거의 모든 대형 AI 모델(GPT, BERT, Claude 등)의 뿌리입니다.
