# 더 나아가기

> 이 자료를 다 보시고 "다음은?" 이라고 물으시는 분께 드리는 가이드입니다.
> 어디로 가야 할지 길이 보이실 거예요.

---

## 다음 단계 트랙

본인의 관심사에 따라 다른 길이 있어요.

```
이 자료 끝남
    ↓
    ├─→ 트랙 1: 이미지 / CV (Computer Vision)
    ├─→ 트랙 2: 텍스트 / NLP (Natural Language)
    ├─→ 트랙 3: 시계열 / 강화학습
    ├─→ 트랙 4: 생성형 AI / LLM
    ├─→ 트랙 5: MLOps / 인프라
    └─→ 트랙 6: 캐글로 실력 키우기
```

각 트랙별로 자세히 봅시다.

---

## 트랙 1: 이미지 / CV

이미지를 다루는 AI를 배우고 싶으시면.

### 배워야 할 것

1. **CNN (Convolutional Neural Networks)**
   - Conv2d, MaxPool, Padding
   - LeNet → AlexNet → VGG → ResNet
   - Transfer Learning (사전학습 모델 활용)

2. **현대 비전 모델**
   - Vision Transformer (ViT)
   - Swin Transformer
   - DINO, MAE (자기지도학습)

3. **태스크별**
   - Object Detection (YOLO, Faster R-CNN, DETR)
   - Segmentation (U-Net, Mask R-CNN, SAM)
   - Image Classification

4. **Diffusion Models** (이미지 생성)
   - Stable Diffusion
   - DALL-E
   - ControlNet

### 추천 자료

- **책**: "Deep Learning with PyTorch" (Eli Stevens)
- **강의**: Stanford CS231n
- **실습**: Kaggle CV 대회

### 핵심 라이브러리
- `torchvision`
- `timm` (사전학습 모델 모음)
- `albumentations` (데이터 증강)
- `segmentation-models-pytorch`

---

## 트랙 2: 텍스트 / NLP

언어를 다루는 AI.

### 배워야 할 것

1. **NLP 기초**
   - Tokenization (BPE, WordPiece)
   - Word Embeddings (Word2Vec, GloVe — 옛날 거지만 알아두면 좋음)

2. **Transformer**
   - Attention Mechanism
   - Self-Attention
   - Encoder vs Decoder

3. **Pre-trained Models**
   - BERT (Encoder)
   - GPT (Decoder)
   - T5 (Encoder-Decoder)

4. **현대 LLM**
   - GPT-4, Claude, Gemini의 구조
   - Fine-tuning (LoRA, QLoRA)
   - RAG (Retrieval-Augmented Generation)
   - Prompting 기법

### 추천 자료

- **책**: "Natural Language Processing with Transformers" (Lewis Tunstall)
- **강의**: Hugging Face NLP Course (무료)
- **사이트**: huggingface.co

### 핵심 라이브러리
- `transformers` (Hugging Face) ⭐
- `datasets`
- `tokenizers`
- `langchain` (LLM 응용)

---

## 트랙 3: 시계열 / 강화학습

### 시계열

매출 예측, 주식, 날씨, 수요 예측 등.

**배워야 할 것:**
- ARIMA, SARIMA (전통적)
- Prophet (Facebook의 도구)
- LSTM, GRU (RNN 계열)
- Temporal Fusion Transformer
- N-BEATS, N-HiTS

**라이브러리:** `statsmodels`, `prophet`, `pytorch-forecasting`

### 강화학습

게임, 로봇, 자율주행, 추천 시스템.

**배워야 할 것:**
- Markov Decision Process
- Q-Learning, Deep Q-Network
- Policy Gradient
- PPO, SAC, A3C

**자료:** "Deep Reinforcement Learning Hands-On" (Maxim Lapan)
**라이브러리:** `stable-baselines3`, `gymnasium`

⚠️ 강화학습은 ML 중 가장 어려워요. 깊이 들어가실 분만 추천.

---

## 트랙 4: 생성형 AI / LLM

ChatGPT 같은 거 만들기.

### LLM 활용 (사용자 입장)

LLM API로 어플 만들기. 가장 빠르게 시작 가능.

**배워야 할 것:**
- API 사용법 (OpenAI, Anthropic, Google)
- Prompt Engineering
- RAG (자료를 검색해서 답하기)
- Function Calling / Tool Use
- Agents (Multi-step reasoning)

**자료:**
- OpenAI / Anthropic 공식 문서
- DeepLearning.AI의 "ChatGPT Prompt Engineering" 강의
- LangChain, LlamaIndex 문서

**라이브러리:**
- `openai` (OpenAI SDK)
- `anthropic` (Claude SDK)
- `langchain`, `llamaindex`
- `chromadb`, `qdrant` (벡터 DB)

### LLM 직접 만들기 / 파인튜닝

자체 모델 만들기. 자원이 많이 필요.

**배워야 할 것:**
- Transformer 아키텍처 깊이
- Fine-tuning (LoRA, QLoRA)
- Distributed Training
- Quantization
- 인스트럭션 튜닝, RLHF

**자료:**
- "Build a Large Language Model" (Sebastian Raschka)
- Hugging Face PEFT 문서

**라이브러리:** `transformers`, `peft`, `trl`, `vllm`

### 이미지 생성

**배워야 할 것:**
- Stable Diffusion 아키텍처
- ControlNet, LoRA for SD
- DreamBooth (특정 인물/객체 생성)

**라이브러리:** `diffusers` (Hugging Face)

---

## 트랙 5: MLOps / 인프라

ML 시스템을 잘 운영하는 일.

### 배워야 할 것

1. **컨테이너 / 오케스트레이션**
   - Docker (이 자료에서 다룸)
   - Kubernetes
   - Helm Charts

2. **워크플로우**
   - Airflow ⭐
   - Prefect, Dagster
   - Kubeflow Pipelines

3. **모델 관리**
   - MLflow ⭐
   - DVC (데이터 버전 관리)
   - Weights & Biases

4. **모니터링**
   - Prometheus + Grafana
   - Evidently AI (데이터 드리프트)
   - Sentry (에러)

5. **클라우드**
   - AWS SageMaker
   - GCP Vertex AI
   - Azure ML

6. **CI/CD**
   - GitHub Actions
   - GitLab CI

### 추천 자료

- **책**: "Designing Machine Learning Systems" (Chip Huyen) ⭐⭐⭐
- **책**: "Machine Learning Engineering" (Andriy Burkov)
- **사이트**: madewithml.com (무료)
- **블로그**: huyenchip.com

---

## 트랙 6: 캐글로 실력 키우기

이론보다 실전. **이 자료 다 보고 가장 추천하는 길.**

### 캐글 시작

1. [kaggle.com](https://kaggle.com) 가입
2. **Getting Started 대회** 참여 (Titanic, House Prices)
3. 다른 사람의 노트북 읽고 베끼기 (Plagiarism 아님, 학습)
4. 본인 노트북 공개

### 점진적 도전

```
1. Getting Started (입문)
   ↓
2. Playground Series (정형 데이터)
   ↓
3. Featured (실제 상금 대회)
   ↓
4. Research (최신 연구 주제)
```

### 메달 시스템

- **Bronze, Silver, Gold** 메달
- Master, Grandmaster 등급

처음부터 메달 기대하지 마세요. **참여 자체가 학습**입니다.

### 추천 대회 종류 (입문자용)

- 정형 데이터 회귀/분류 (XGBoost 위주)
- 이미지 분류 (CNN 입문)
- NLP (Disaster Tweets)

---

## 책 추천 (난이도순)

### 입문
1. **혼자 공부하는 머신러닝+딥러닝** (박해선) — 한국어, 가장 친절
2. **Hands-On Machine Learning** (Aurélien Géron) — 영어/한역, 표준
3. **Deep Learning with PyTorch** (Eli Stevens) — PyTorch 입문

### 중급
4. **Python Machine Learning** (Sebastian Raschka)
5. **Designing Machine Learning Systems** (Chip Huyen) — MLOps 필독서
6. **Build a Large Language Model** (Sebastian Raschka) — LLM

### 고급 / 이론
7. **Pattern Recognition and Machine Learning** (Bishop) — 통계적 ML 바이블
8. **Deep Learning** (Goodfellow, Bengio, Courville) — 딥러닝 바이블, 무료 PDF

---

## 강의 추천

### 한국어
- **모두를 위한 딥러닝** (김성훈) — 무료, 친절 (좀 오래됨)
- **혼공머신** (박해선) — 책의 영상

### 영어
- **Andrew Ng의 ML 시리즈** (Coursera) — 가장 유명
- **fast.ai** (Jeremy Howard) — 실전 위주, 무료
- **DeepLearning.AI 시리즈** — 다양한 주제, 짧고 좋음
- **CS231n** (Stanford, CV) — 무료 영상
- **CS224n** (Stanford, NLP) — 무료 영상

---

## 따라가야 할 사람들 (X / Twitter)

ML 분야의 변화는 빠릅니다. 트위터로 따라잡기.

- **Andrej Karpathy** (전 Tesla AI 리더) - LLM 강의 영상도
- **Yann LeCun** (Meta Chief AI) - 컴퓨터 비전 거장
- **Andrew Ng** - ML 교육의 전설
- **Sebastian Raschka** - PyTorch 좋은 글 많이 씀
- **Jeremy Howard** - fast.ai 창립자

한국:
- **유재준** (신경망과 딥러닝)
- **하용호** (시계열, AB 테스트)

---

## 커뮤니티

질문하고 답변 받기 좋은 곳들.

### 한국
- **OKKY** — 개발자 커뮤니티
- **TensorFlow KR / PyTorch KR** — 페이스북 그룹
- **AI 커뮤니티 카카오톡** — 검색해서 찾기
- **GeekNews / 디스코드** — 일부 ML 채널

### 영어
- **Reddit /r/MachineLearning** — 학계 글 많음
- **Hacker News** — 산업 트렌드
- **Discord** — 거의 모든 라이브러리가 디스코드 있음
- **Stack Overflow** — 코드 디버깅

---

## 본인 프로젝트 시작하기

가장 좋은 학습은 **본인의 관심사로 만든 프로젝트**.

### 아이디어 예시

#### 데이터 수집 + 분석
- 본인 카페에 가는 지출 분석
- 즐겨 보는 유튜브 채널 통계
- 본인 메일 카테고리 분류기

#### NLP
- 이메일 자동 분류
- 일기 감정 분석
- 챗봇 (간단한 것부터)

#### CV
- 본인의 식단 사진 분류
- 손글씨 메모 → 텍스트
- 강아지 종 식별

#### 추천
- 영화 추천
- 책 추천
- 음식 추천

작게 시작하세요. 완성하지 못해도 OK. **시작이 전부**입니다.

---

## 마지막 조언

1. **꾸준함이 천재를 이깁니다.** 매일 30분이라도.
2. **막히면 잠시 떠나세요.** 산책, 잠. 다음 날 풀려요.
3. **남과 비교하지 마세요.** 본인의 어제와만 비교.
4. **나누세요.** 배운 걸 블로그에 적으면 본인이 더 배웁니다.
5. **즐기세요.** 재미가 있어야 오래 가요.

---

ML은 끝없이 배울 게 많은 분야예요. 평생 학습이라는 말이 정말 맞아요. 그래서 더 매력적이기도 하고요.

**여러분의 ML 여정을 응원합니다.** 좋은 길 가십시오.

---

**감사의 말**

이 자료를 끝까지 읽어주셔서 진심으로 감사합니다.
이 자료가 여러분의 다음 단계로 가는 작은 디딤돌이 되기를 바랍니다.

➡️ [메인으로 돌아가기](../README.md)
