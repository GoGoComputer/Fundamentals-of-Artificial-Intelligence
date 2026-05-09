# 7.4 Transformer · Attention — 모든 LLM의 뿌리

## 들어가며

2017년 구글 연구진이 발표한 「Attention Is All You Need」는 AI 역사의 분기점이었습니다. 이 논문에서 제안한 **Transformer** 구조는 당시 NLP를 지배하던 RNN/LSTM을 5년 안에 거의 완전히 대체했습니다.

지금 우리가 매일 쓰는 ChatGPT, Claude, Gemini, BERT, Whisper, Stable Diffusion까지 — 거의 모든 대형 AI 모델이 Transformer를 기반으로 합니다.

뭐가 그렇게 특별했을까요? 한 줄로 요약하면 이렇습니다.

> RNN은 시퀀스를 한 토큰씩 순차적으로 처리합니다. Transformer는 전체 시퀀스를 한 번에, 병렬로 처리합니다.

이 차이가 GPU 시대에 모든 것을 뒤바꿨습니다. 학습이 10배, 100배 빨라졌고, 더 큰 모델을 만들 수 있게 되었으며, 그 결과가 오늘날의 LLM입니다.

> 🧮 **수학 보충** — 이번 장의 핵심 연산은 행렬 곱입니다. [부록/수학/03-선형대수.md](../../부록/수학/03-선형대수.md)의 내적과 [03a-행렬-깊게.md](../../부록/수학/03a-행렬-깊게.md)의 행렬 곱셈을 떠올리며 따라오시면 됩니다.

---

## 1. Attention의 직관 — 모든 곳을 동시에 보기

### 1.1 RNN의 한계 다시 짚기

RNN은 정보를 한 줄짜리 hidden state로 압축해 다음 시점에 넘깁니다. 긴 문장에서 앞쪽 정보가 뒤로 갈수록 흐려지죠.

```
"나는 어렸을 때 [프랑스]에서 살았다. 그래서 ___로 꿈을 꾼다."
        ↑                                  ↑
        여기서 본 정보가              여기까지 살아남아야 함
```

### 1.2 Attention의 발상

Attention은 이 문제를 정면 돌파합니다.

> "기억을 하나로 압축하지 말고, **모든 단어를 그대로 두고**, 필요할 때마다 관련 있는 단어를 직접 꺼내서 보자."

빈칸을 예측할 때 모델은 "프랑스"라는 단어가 30단어 떨어져 있어도 그 위치로 직접 가서 정보를 가져옵니다. 메모를 다시 쓸 필요 없이 책 전체를 펼쳐놓고 필요한 페이지를 찾아보는 셈입니다.

### 1.3 Query, Key, Value — 도서관 비유

Attention의 핵심 메커니즘을 도서관 비유로 풀어봅니다.

- **Query (Q)**: "내가 찾고 있는 것" — 검색어
- **Key (K)**: "각 책의 색인 태그" — 책마다 붙어있는 키워드
- **Value (V)**: "책의 실제 내용" — 진짜 정보

작동 방식:

1. Query와 모든 Key의 유사도를 계산 → "어느 책이 내 검색어와 가장 비슷한가"
2. 유사도를 0~1 사이 가중치로 변환 (softmax)
3. 모든 Value를 그 가중치로 평균 → "관련 있는 책 내용을 비중 있게 종합"

수식으로는 그 유명한

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

`√d_k`로 나누는 이유는 행렬 곱의 결과 분산이 차원에 따라 커지는 것을 안정시키기 위해서입니다.

### 1.4 Self-Attention — 자기가 자기에게 묻기

Self-Attention은 같은 문장 안에서 Q, K, V를 모두 만듭니다. 즉 "내 문장의 각 단어가 같은 문장의 다른 단어들에게 얼마나 주목해야 하나"를 계산합니다.

```
"The animal didn't cross the street because it was too tired"

"it"이라는 단어를 처리할 때:
  → "animal"에 70% 주목
  → "street"에 20% 주목
  → 나머지 단어들에 10% 분산
```

문법 규칙을 안 가르쳐줘도 모델이 데이터에서 이런 의존 관계를 스스로 발견합니다.

---

## 2. Multi-Head Attention — 여러 시각에서 보기

한 번의 어텐션은 한 가지 관점만 잡습니다. 그래서 Transformer는 어텐션을 여러 개(보통 8개, 12개, 16개) 병렬로 돌립니다.

각 head는 서로 다른 측면을 학습합니다.
- Head 1: 주어-동사 관계
- Head 2: 수식 관계 (형용사-명사)
- Head 3: 거리 의존성
- ...

```python
import torch
import torch.nn as nn

# embed_dim=512, head 개수 8개 → 각 head는 64차원 처리
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

x = torch.randn(2, 10, 512)   # (배치, 시퀀스, 차원)
out, weights = mha(x, x, x)   # Q=K=V=x → self-attention

print(out.shape)      # torch.Size([2, 10, 512])
print(weights.shape)  # torch.Size([2, 10, 10]) — 어텐션 가중치 행렬
```

`weights`를 시각화하면 모델이 어떤 단어에 주목했는지 보입니다. 해석 가능성 측면에서 RNN보다 큰 장점이죠.

---

## 3. Positional Encoding — 순서 정보 주입하기

Attention은 모든 단어를 한꺼번에 보기 때문에, 그 자체로는 순서를 모릅니다. "나 너 사랑해"와 "너 나 사랑해"를 같은 입력으로 봅니다.

이를 해결하려고 각 위치마다 고유한 벡터를 더해줍니다.

### 3.1 사인/코사인 방식 (원조 Transformer)

원 논문은 위치 `pos`와 차원 `i`에 대해 다음 식을 씁니다.

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

장점은 학습 데이터에 없던 더 긴 문장에도 외삽 가능하다는 점입니다.

### 3.2 학습 가능한 위치 임베딩 (BERT, GPT 등)

요즘은 위치 임베딩 자체를 가중치로 두고 학습시키는 방식이 더 흔합니다.

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        positions = torch.arange(L, device=x.device)
        return x + self.pos_emb(positions)
```

### 3.3 RoPE — 요즘 LLM이 쓰는 방식

LLaMA, GPT-NeoX 등 최신 LLM은 **RoPE(Rotary Positional Embedding)**를 씁니다. Query와 Key 벡터를 위치에 따라 회전시키는 방식이라, 상대적 위치가 자연스럽게 표현됩니다. 이 책 범위는 넘어가지만, 이름은 알아두세요.

---

## 4. Transformer 블록 전체 구조

이제 부품들을 조립해봅니다.

```
입력 (B, L, d)
    ↓
[LayerNorm] → [Multi-Head Attention] → (+) ← residual
    ↓
[LayerNorm] → [Feed-Forward (MLP)] → (+) ← residual
    ↓
출력 (B, L, d)
```

핵심 부품 4가지:
1. **Multi-Head Attention** — 단어들 간 관계 학습
2. **Feed-Forward Network** — 각 위치마다 독립적인 MLP (보통 차원을 4배로 늘렸다 줄임)
3. **Residual Connection** — ResNet과 같은 skip connection
4. **Layer Normalization** — Batch Norm 대신 토큰별로 정규화

이런 블록을 N개 쌓으면 Transformer 인코더가 됩니다. BERT-base는 12개, GPT-3는 96개를 쌓았습니다.

### 4.1 PyTorch로 직접 만들기

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN 방식 (요즘 표준)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + self.dropout(x)

        return x

# 6층짜리 인코더 만들기
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len=512, d_model=512,
                 num_heads=8, num_layers=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        L = x.size(1)
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

model = Encoder(vocab_size=10000)
x = torch.randint(0, 10000, (2, 50))
print(model(x).shape)   # torch.Size([2, 50, 512])
```

이 100줄짜리 코드가 BERT의 뼈대와 거의 같습니다.

---

## 5. Encoder vs Decoder vs Encoder-Decoder

Transformer는 사용 방식에 따라 세 종류로 나뉩니다.

### 5.1 Encoder-only — 이해 모델

대표: **BERT, RoBERTa, DeBERTa**

문장 전체를 양방향으로 봅니다. 분류, 개체명 인식, 질의응답 등 "이해"가 필요한 작업에 강합니다.

```
입력: "이 영화 정말 [MASK]"
   ↓ [Encoder]
출력: 모든 단어의 의미 벡터 → 분류기로 [MASK] = "재밌었다" 예측
```

### 5.2 Decoder-only — 생성 모델

대표: **GPT 계열, LLaMA, Claude, Gemini**

이전 단어들만 보고 다음 단어를 예측합니다(causal masking). 생성에 특화되어 있습니다.

```
입력: "옛날 옛적에"
   ↓ [Decoder, 다음 단어 예측 반복]
출력: "옛날 옛적에 깊은 산속에 호랑이가 살았는데..."
```

요즘 LLM은 거의 다 이 구조입니다.

### 5.3 Encoder-Decoder — 변환 모델

대표: **T5, BART, 원조 Transformer**

인코더가 입력을 이해하고, 디코더가 출력을 생성합니다. 번역, 요약처럼 입력과 출력이 다른 시퀀스인 작업에 적합합니다.

```
입력 (한국어): "나는 학교에 간다"
   ↓ [Encoder]
의미 벡터
   ↓ [Decoder, cross-attention으로 인코더 출력 참조]
출력 (영어): "I go to school"
```

---

## 6. Causal Mask — 미래를 못 보게 하기

Decoder를 학습할 때 문제가 있습니다. "옛날 옛적에 ___"의 빈칸을 예측해야 하는데, Self-Attention은 모든 단어를 한 번에 보니 정답("호랑이가")까지 봐버립니다.

이를 막으려면 **causal mask(인과 마스크)**를 씁니다. 어텐션 행렬의 우상단 삼각형을 −∞로 만들어, 미래 위치를 못 보게 차단합니다.

```python
def causal_mask(L):
    # 위쪽 삼각형을 1로 (마스킹할 위치)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    return mask

mask = causal_mask(5)
print(mask.int())
# tensor([[0, 1, 1, 1, 1],
#         [0, 0, 1, 1, 1],
#         [0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0]])
```

이 마스크를 Multi-Head Attention의 `attn_mask`에 넘기면 됩니다. 1이 있는 위치는 무시됩니다.

---

## 7. 사전학습과 파인튜닝 — LLM이 만들어지는 방식

### 7.1 사전학습 (Pretraining)

엄청난 양의 텍스트로 단순한 목표를 학습합니다.

- **BERT 방식 (Masked Language Modeling)**: 문장의 15% 단어를 가리고 맞히기
- **GPT 방식 (Causal Language Modeling)**: 다음 단어 예측

이 단순한 작업을 수십~수천억 토큰으로 학습하면, 모델이 문법, 사실 지식, 추론 능력까지 함께 익힙니다. 라벨이 필요 없는 self-supervised learning이라 데이터 양이 핵심입니다.

### 7.2 파인튜닝 (Fine-tuning)

사전학습된 모델을 특정 작업용으로 조정합니다.

- **분류 파인튜닝**: BERT 위에 분류기 붙여 감성 분석
- **지시 학습 (Instruction Tuning)**: GPT를 "지시-답변" 쌍으로 학습 → ChatGPT
- **RLHF**: 사람 피드백으로 강화학습 → 더 자연스러운 응답
- **LoRA, QLoRA**: 일부 가중치만 학습해 메모리 절약

```python
# Hugging Face transformers로 BERT 파인튜닝 (감성 분석)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

inputs = tokenizer("This movie was great!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)  # torch.Size([1, 2])
```

이렇게 5줄이면 사전학습된 BERT 위에 분류 헤드가 자동으로 붙습니다.

---

## 8. 실전 예제 — 사전학습된 BERT로 감성 분석 파인튜닝

```python
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset

# --- 1. 데이터와 토크나이저 ---
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True,
                    padding="max_length", max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(2000)),
                         batch_size=16, shuffle=True)
test_loader = DataLoader(dataset["test"].select(range(1000)), batch_size=16)

# --- 2. 모델 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

# --- 3. 옵티마이저와 스케줄러 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * num_steps),
    num_training_steps=num_steps
)

# --- 4. 학습 ---
for epoch in range(3):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       labels=batch["label"])
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 평가
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"]).logits.argmax(-1)
            correct += (preds == batch["label"]).sum().item()
    print(f"Epoch {epoch+1} | acc {correct/1000:.4f}")
```

이 정도면 보통 3 epoch 만에 90% 이상 정확도가 나옵니다. LSTM 베이스라인(85~89%)을 가볍게 넘습니다.

### 8.1 핵심 포인트

1. **AdamW + warmup** — Transformer는 학습률 워밍업이 거의 필수입니다. 처음 10%는 학습률을 0에서 점진적으로 올립니다.
2. **작은 학습률 (2e-5)** — 사전학습 가중치를 망치지 않으려면 1e-5~5e-5 범위.
3. **Truncation** — BERT는 최대 512 토큰까지만 처리. 긴 문장은 잘라야 합니다.

---

## 9. Transformer를 쓸 때 알아둘 점

### 9.1 메모리 사용량은 시퀀스 길이의 제곱

어텐션 행렬이 `L × L`이라 시퀀스가 길어질수록 메모리가 폭발합니다.

```
시퀀스 1024 → 어텐션 행렬 1M개
시퀀스 4096 → 16M개
시퀀스 32768 → 1B개 (대부분 GPU 못 버팀)
```

이 문제를 풀려고 Flash Attention, Sliding Window Attention, Linear Attention 같은 변형이 계속 나오고 있습니다.

### 9.2 작은 데이터셋엔 안 좋다

Transformer는 데이터가 많을 때 진가를 발휘합니다. 1만 건 정도의 작은 데이터셋이면 사전학습 모델 파인튜닝이 정답입니다. 처음부터 학습하지 마세요.

### 9.3 Tokenization이 중요

LLM의 입출력은 단어가 아니라 **토큰**입니다. BPE(Byte-Pair Encoding), WordPiece, SentencePiece 같은 알고리즘이 자주 쓰는 부분 단어를 추출합니다.

```python
tokenizer.tokenize("unbelievable")
# ['un', '##believ', '##able']  — WordPiece 예시
```

토큰화 방식이 다르면 같은 문장이라도 처리되는 단위가 달라집니다.

### 9.4 환각(Hallucination)

LLM은 그럴듯한 거짓말을 자주 합니다. 사실 검증이 필요한 답변에는 RAG(Retrieval-Augmented Generation)나 외부 도구 사용을 결합해야 합니다.

---

## 10. 한눈에 정리

| 개념 | 한 줄 요약 |
|------|-----------|
| Self-Attention | 같은 시퀀스 내 단어들 간 관계 계산 |
| Q, K, V | 검색어, 색인, 내용 |
| Multi-Head | 여러 관점에서 동시에 어텐션 |
| Positional Encoding | 순서 정보 주입 (sin/cos, 학습형, RoPE) |
| Layer Norm | 토큰별 정규화 (BatchNorm 대신) |
| Causal Mask | 디코더가 미래 못 보게 차단 |
| Encoder-only | BERT 류, 이해 작업 |
| Decoder-only | GPT 류, 생성 작업 |
| Encoder-Decoder | T5 류, 변환 작업 |

---

## 더 깊이 가고 싶으시면

- **Vision Transformer (ViT)**: 이미지를 패치로 잘라 Transformer에 넣기
- **CLIP**: 이미지와 텍스트를 같은 임베딩 공간에
- **Diffusion + Transformer**: Stable Diffusion 3, DiT 등 이미지 생성
- **MoE (Mixture of Experts)**: 거대 모델을 효율적으로. Mixtral, Gemini Pro
- **Flash Attention**: 메모리 효율적 어텐션 구현
- **장문 처리**: Longformer, BigBird, Mamba (Transformer 대안)
- **Hugging Face 생태계**: `transformers`, `datasets`, `accelerate` 라이브러리

다음 장에서는 모델 학습 전 단계인 **EDA(탐색적 데이터 분석)와 특성 공학**을 다룹니다. "garbage in, garbage out" — 어떤 모델을 쓰든 데이터 품질이 결과를 결정합니다.
