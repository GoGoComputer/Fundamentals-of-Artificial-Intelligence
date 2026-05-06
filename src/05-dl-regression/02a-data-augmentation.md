# 5.2a (보강) 데이터 증강 (Data Augmentation)

## 가장 강력한 과적합 방지법

[5.2 과적합이란](02-overfitting.md)의 끝에서 이렇게 말씀드렸어요.

> "**데이터를 더 모으는 게 가장 강력한 과적합 방지**입니다."

근데 데이터를 더 모으는 건 비싸요. **그럼 가짜로 늘리면 어떨까요?**

이게 **데이터 증강 (Data Augmentation)** 입니다. 기존 데이터를 살짝 변형해서 "새로운 샘플"을 만드는 거예요.

---

## 왜 효과가 있어요?

### 비유

영어 시험 공부를 한다고 해 봐요.

**나쁜 공부:** 한 단어집을 100번 보기 → 단어 외움. 다른 단어집의 단어는 모름.

**좋은 공부:** 같은 단어를 여러 책에서 보기. 같은 단어가 다른 문맥에서 어떻게 쓰이는지.

데이터 증강도 같아요. **같은 본질을 다른 모습으로** 모델에게 보여주면, 모델이 진짜 패턴을 학습하게 돼요. 표면적인 노이즈는 외우지 않고요.

---

## 분야별 증강 기법

### 이미지 증강 (가장 발달)

이미지는 변형해도 의미가 안 변해요. 강아지 사진을 회전시켜도 강아지죠.

**자주 쓰는 변형들:**

| 변형 | 설명 |
|------|------|
| **Horizontal Flip** | 좌우 반전 |
| **Vertical Flip** | 상하 반전 (적절할 때만) |
| **Rotation** | 회전 (예: ±15°) |
| **Crop** | 일부만 잘라내기 |
| **Resize** | 크기 변경 |
| **Color Jitter** | 색상/명도/대비 변경 |
| **Gaussian Noise** | 노이즈 추가 |
| **Cutout** | 일부 가리기 |
| **Mixup** | 두 이미지를 섞기 |
| **CutMix** | 두 이미지의 일부를 합치기 |

```python
# torchvision의 transforms
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# 평가 시엔 증강 안 함
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(...),
])
```

### 더 강력한 라이브러리: Albumentations

torchvision보다 빠르고 변형이 많음. 객체 탐지, 세그멘테이션에 강해요.

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(p=0.3),    # Cutout
    A.Normalize(),
])
```

---

### 텍스트 증강

이미지보다 어려움. 단어 하나만 바뀌어도 의미가 달라질 수 있거든요.

**자주 쓰는 기법:**

| 기법 | 예시 |
|------|------|
| **Synonym Replacement** | "큰 차" → "거대한 차" |
| **Random Insertion** | 무작위 단어 삽입 |
| **Random Swap** | 단어 순서 바꾸기 |
| **Random Deletion** | 단어 삭제 |
| **Back Translation** | 한→영→한 번역 |
| **Token Masking** | 일부 단어 마스크 (BERT 학습법) |

```python
# nlpaug 라이브러리
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_p=0.3)
augmented = aug.augment("이 영화는 정말 재미있어요")
```

---

### 표 데이터 증강 (어려움)

표 데이터는 증강이 가장 어려워요. 의미를 보존하기가 까다롭거든요.

**자주 쓰는 기법:**

| 기법 | 설명 |
|------|------|
| **SMOTE** | 소수 클래스 샘플의 중간점 생성 (불균형 해결) |
| **GANs** | GAN으로 가짜 샘플 생성 |
| **Noise 추가** | 수치 특성에 작은 노이즈 |
| **Feature Engineering** | 특성 조합 (사실상 증강 아님) |

```python
# SMOTE 사용
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

### 시계열 증강

| 기법 | 설명 |
|------|------|
| **Time Warping** | 시간 축 늘이기/줄이기 |
| **Magnitude Warping** | 값 크기 변경 |
| **Window Slicing** | 일부 구간만 사용 |
| **Jitter** | 노이즈 추가 |
| **Permutation** | 구간 순서 변경 |

---

## 실전: MNIST에 증강 적용

신경망 분류 코드에 추가만 하면 돼요.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습용 (증강 O)
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),     # 약간만 회전
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 위치 살짝
    transforms.ToTensor(),
])

# 평가용 (증강 X)
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

⚠️ **MNIST에서 좌우 반전은 안 됨** ('6'이 '9'가 됨). 도메인 지식 필요.

---

## 효과 — 진짜로 일어나는 일

같은 모델, 같은 학습 코드로 증강 있고 없고 비교:

| | 증강 없음 | 증강 있음 |
|---|---|---|
| Train Acc | 99.5% | 98.0% |
| Test Acc | 96.2% | **98.5%** |
| 과적합 | 심함 | 거의 없음 |

Train 점수는 약간 떨어지지만 (학습이 더 어려워짐), **Test 점수가 올라가요.** 이게 진짜 일반화.

---

## 함정

### 함정 1: 평가 시에도 증강

```python
# ❌ 잘못
test_set = datasets.MNIST(..., transform=train_transform)   # 증강 적용

# ✅ 올바름
test_set = datasets.MNIST(..., transform=test_transform)   # 증강 X
```

평가는 항상 원본 데이터로.

### 함정 2: 의미를 바꾸는 증강

- MNIST에 좌우 반전: 6/9 헷갈림
- 의료 영상에 색상 변경: 진단에 영향
- 차량 인식에 비현실적 회전 (90도+)

**도메인 지식 필수.** 적용 전에 "이 변형이 본질을 보존하는가?" 묻기.

### 함정 3: 너무 강한 증강

```python
# ❌ 너무 강함
transforms.RandomRotation(degrees=180)   # 글씨가 거꾸로
transforms.ColorJitter(brightness=2.0)   # 거의 안 보임
transforms.GaussianNoise(std=1.0)        # 원본 소실

# ✅ 적당
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(brightness=0.2)
transforms.GaussianNoise(std=0.05)
```

증강이 너무 강하면 학습 데이터의 분포가 진짜 데이터의 분포에서 멀어져요.

### 함정 4: 정규화 순서

```python
# ❌ 잘못 — Normalize 후에 변형하면 분포 깨짐
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomRotation(15),     # ← 정규화 후
])

# ✅ 올바름 — 변형 다 한 후에 정규화
transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),    # ← 마지막에
])
```

---

## 고급 기법: Mixup

두 샘플을 **선형으로 섞어서** 학습.

```python
# 두 샘플 (x1, y1), (x2, y2)
lambda_ = 0.7   # 0~1 사이 무작위
x_mixed = lambda_ * x1 + (1 - lambda_) * x2
y_mixed = lambda_ * y1 + (1 - lambda_) * y2
```

이미지 두 장을 살짝 합친 느낌. 학습 라벨도 같이 섞음.

```python
def mixup(x, y, alpha=1.0):
    lambda_ = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    
    x_mixed = lambda_ * x + (1 - lambda_) * x[idx]
    y_a, y_b = y, y[idx]
    
    return x_mixed, y_a, y_b, lambda_


# 학습 루프 안에서
x, y = batch
x_mixed, y_a, y_b, lam = mixup(x, y)

output = model(x_mixed)
loss = lam * loss_fn(output, y_a) + (1 - lam) * loss_fn(output, y_b)
```

이미지 분류에서 효과 크게 입증됨. CutMix도 비슷한 아이디어.

---

## 정리

```python
# 학습용 (증강)
train_transform = transforms.Compose([
    # 데이터 적합한 변형들
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 평가용 (증강 X)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

**한 줄 요약: "데이터를 가짜로 늘려서 모델이 일반화하게 만든다."**

**효과 큰 분야:**
1. 이미지 (가장 효과 큼) ✅✅✅
2. 텍스트 (효과 보통) ✅✅
3. 음성 (효과 큼) ✅✅✅
4. 표 데이터 (어려움) ✅
5. 시계열 (효과 보통) ✅✅

---

## 자주 묻는 질문

> **Q. 데이터 증강만으로 충분해요?**
>
> 아니요. **다른 정규화 (Dropout, L2 등)와 같이** 쓰는 게 보통. 진짜 데이터를 더 모으는 게 항상 가장 좋음.

> **Q. 증강을 얼마나 강하게 해야 해요?**
>
> 검증 점수를 보고 결정. 약하게 시작해서 점점 강하게. Val Loss가 떨어지면 좋은 신호.

> **Q. CutMix와 Mixup 중 뭐가 좋아요?**
>
> 이미지 분류에서 둘 다 자주 씀. 데이터에 따라 다르므로 둘 다 시도.

➡️ 이전: [5.2 과적합이란](02-overfitting.md)
➡️ 다음: [5.3 Early Stopping](03-early-stopping.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **5.2a (보강) 데이터 증강 (Data Augmentation)** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **5.2a (보강) 데이터 증강 (Data Augmentation)**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "5.2a (보강) 데이터 증강 (Data Augmentation) 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **5.2a (보강) 데이터 증강 (Data Augmentation)**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

5.2a (보강) 데이터 증강 (Data Augmentation)는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
