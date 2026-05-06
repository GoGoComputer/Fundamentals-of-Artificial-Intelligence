# 2.8a (보강) 클래스 불균형 다루기

> 본문 [2.5 평가 지표](05-evaluation-metrics.md)의 "정확도 99%인데 모델이 쓰레기" 사례를 기억하시나요?
> 그게 **클래스 불균형(Class Imbalance)** 문제예요. 이 글은 그 해결법.

---

## 문제 상황 다시 보기

| 데이터 분포 | 모델이 무조건 "정상"으로 답하면? |
|-----------|----------------------------|
| 정상 99명, 환자 1명 | 정확도 99%, 환자 0명 발견 ❌ |
| 정상 999명, 사기 1건 | 정확도 99.9%, 사기 0건 적발 ❌ |
| 정상 9999개, 불량품 1개 | 정확도 99.99%, 불량 0개 발견 ❌ |

이런 데이터에 일반 학습을 그대로 적용하면, 모델이 **다수 클래스만 답하는 게 가장 쉬운 길**이에요. 그게 손실이 가장 작거든요.

소수 클래스를 잡아내야 가치 있는 일이 많아요.
- 의료 진단 (병에 걸린 환자 찾기)
- 사기 탐지 (수상한 거래)
- 불량품 검출 (제조)
- 고객 이탈 예측 (떠날 사람 미리 알기)

---

## 해결 방법 4가지

### 방법 1: 평가 지표를 바꾸기

가장 먼저 할 일. **정확도(accuracy) 보지 마세요.** 대신:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# precision, recall, F1 다 봄
```

특히:
- **Recall (재현율)**: 진짜 환자 중 모델이 잡은 비율 → **놓치면 안 되는 상황에서**
- **Precision (정밀도)**: 환자라고 한 사람 중 진짜 환자 비율 → **헛소리 방지**
- **F1**: 둘의 조화 평균
- **PR-AUC**: 불균형에서 ROC-AUC보다 더 의미 있음

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score,    # PR-AUC
)

print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
```

### 방법 2: 클래스 가중치 (Class Weight)

손실 계산 시 **소수 클래스의 페널티를 더 크게.**

```python
# sklearn — 한 줄로 끝
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

model = LogisticRegression(class_weight='balanced')
model = SVC(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')
```

`class_weight='balanced'` 가 자동으로 클래스 비율의 역수만큼 가중치 부여.

```python
# 또는 직접 지정
model = SVC(class_weight={0: 1, 1: 10})    # 클래스 1을 10배 무겁게
```

PyTorch에서는:

```python
weights = torch.tensor([1.0, 10.0])    # 클래스 1을 10배
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
```

이게 **가장 쉽고 거의 항상 효과 있는 방법**이에요. 첫 시도로 추천.

### 방법 3: 샘플링 기법

데이터 자체의 분포를 바꿔요.

#### 3-1. Oversampling — 소수 클래스를 늘리기

```python
# 단순한 방법 — 소수 클래스를 복사
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

장점: 정보 손실 X
단점: 같은 샘플 반복 → 과적합 위험

#### 3-2. SMOTE — 소수 클래스를 똑똑하게 늘리기

`SMOTE` (Synthetic Minority Over-sampling Technique) — 소수 클래스 샘플들 사이를 보간해서 **새로운 가짜 샘플** 생성.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

원리:
```
원본:  [● ●]      ←  소수 클래스 2개
       
SMOTE: [● ● ◯ ◯ ◯]   ← 두 점 사이를 보간한 가짜 샘플 추가
       (◯ 가 가짜)
```

장점: 단순 복사보다 다양성 증가
단점: 노이즈도 같이 증폭, 고차원에서는 덜 효과

#### 3-3. Undersampling — 다수 클래스를 줄이기

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

장점: 학습 빠름 (데이터 줄어듦)
단점: **정보 손실 큼** (다수 클래스의 좋은 샘플도 버림)

#### 3-4. SMOTETomek — 위 둘 결합

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

소수 클래스 늘리고, 경계의 잡음 다수 샘플 제거. 균형 잘 잡혀요.

#### 설치

```bash
pip install imbalanced-learn
```

`imblearn`은 sklearn의 자매 라이브러리예요.

### 방법 4: 임계값 (Threshold) 조정

분류 모델의 출력은 **확률**이에요. 기본은 0.5에서 끊지만, 바꿀 수 있어요.

```python
# 기본 (0.5 기준)
y_pred = model.predict(X_test)

# 직접 임계값 조정 (0.3 기준)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)
```

#### 영향

```
임계값 ↓ (예: 0.3) → 더 많이 양성으로 판정
                    → recall ↑, precision ↓

임계값 ↑ (예: 0.7) → 더 적게 양성으로 판정
                    → recall ↓, precision ↑
```

암 진단처럼 **놓치면 안 되는 상황**에서는 임계값을 낮춰서 recall을 올려요. 필요한 만큼.

#### 최적 임계값 찾기

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1 가장 큰 임계값
f1_scores = 2 * precision * recall / (precision + recall)
best_idx = f1_scores.argmax()
print(f"최적 임계값: {thresholds[best_idx]:.4f}")
print(f"그 때 F1: {f1_scores[best_idx]:.4f}")
```

---

## 어느 방법을 써야 해요?

상황별 추천 순서:

```
1. 평가 지표를 먼저 바꿈 (precision, recall, F1)
   ↓
2. class_weight='balanced' 시도
   ↓
3. SMOTE 또는 RandomOverSampler 시도
   ↓
4. 임계값 조정 (배포 단계)
   ↓
5. 그래도 안 되면 → 데이터 더 모으기 (소수 클래스)
```

**가장 효과적인 건 데이터를 더 모으는 것**이지만, 어려우니 다른 방법으로 보완.

---

## 함정과 주의사항

### 함정 1: Test 데이터에는 절대 SMOTE X

```python
# ❌ 잘못
smote = SMOTE()
X_all_resampled, y_all_resampled = smote.fit_resample(X_full, y_full)
X_train, X_test = train_test_split(X_all_resampled, y_all_resampled)

# ✅ 올바름
X_train, X_test, y_train, y_test = train_test_split(X, y)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Test는 원본 그대로!
```

평가는 **실제 분포**에서 해야 의미 있어요.

### 함정 2: Cross-Validation 시 stratify

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 각 fold가 같은 클래스 비율 유지
```

일반 KFold 쓰면 어떤 fold에는 소수 클래스가 0개일 수 있어요.

### 함정 3: 한 가지 방법만으로 안 됨

보통 **여러 방법을 조합**해야 해요.

```python
# 흔한 조합
model = LogisticRegression(class_weight='balanced')
# + SMOTE
# + 평가 시 F1 보기
# + 임계값 조정
```

### 함정 4: 진짜로 불균형이 문제가 아닐 수도

가끔은 모델이 그냥 못해서 그런 거예요. 베이스라인 모델을 먼저 시도하고, 진짜로 불균형 때문에 망하는지 확인.

---

## 실전 예시 — 신용카드 사기 탐지

신용카드 거래 데이터의 0.17%만 사기인 상황 (매우 불균형).

```python
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# 데이터 로드 (예시)
# X, y = load_credit_fraud_data()

# Stratified 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)


# 베이스라인 (불균형 무시)
print("[베이스라인]")
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)
print(classification_report(y_test, y_pred))


# 방법 1: class_weight
print("\n[Class Weight]")
weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
weighted.fit(X_train, y_train)
y_pred = weighted.predict(X_test)
print(classification_report(y_test, y_pred))


# 방법 2: SMOTE
print("\n[SMOTE]")
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


# PR-AUC (불균형의 표준 지표)
y_proba = pipeline.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_proba)
print(f"\nPR-AUC: {pr_auc:.4f}")
```

전형적 결과:

| 모델 | Recall (사기) | Precision | F1 |
|------|--------------|-----------|-----|
| 베이스라인 | 0.55 | 0.85 | 0.67 |
| Class Weight | **0.83** | 0.42 | 0.56 |
| SMOTE | 0.78 | 0.51 | 0.62 |

**Recall이 크게 올라가요.** 사기를 더 잘 잡아내요. Precision이 낮아지는 건 트레이드오프.

---

## 정리

```python
# 1. 평가 지표
from sklearn.metrics import classification_report, average_precision_score
print(classification_report(y_test, y_pred))    # F1, recall, precision

# 2. Class weight
model = LogisticRegression(class_weight='balanced')

# 3. SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 4. 임계값 조정
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

# 5. Stratified CV
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True)
```

**한 줄 요약:** "정확도 보지 말고, F1과 recall 보세요. class_weight='balanced' 부터 시작하세요."

---

## 자주 묻는 질문

> **Q. 데이터가 95:5 정도여도 불균형 처리해야 해요?**
>
> 보통 80:20 정도까지는 일반 학습으로 충분. 90:10부터 의심, 95:5 이상이면 처리 권장.

> **Q. SMOTE를 쓰면 항상 좋아져요?**
>
> 아닙니다. 노이즈가 많은 데이터에서는 노이즈도 같이 증폭돼서 더 나빠질 수 있어요. 항상 검증하세요.

> **Q. 딥러닝에서도 같은 방법을 쓰나요?**
>
> 네. PyTorch에서는 `nn.CrossEntropyLoss(weight=...)` 또는 `WeightedRandomSampler`. SMOTE는 표 데이터에서만 효과적이고 이미지/텍스트에는 부적합.

> **Q. Focal Loss는 뭔가요?**
>
> 객체 탐지에서 자주 쓰는 손실. 쉬운 샘플의 가중치를 자동으로 줄여요. 다수가 쉬운 샘플인 불균형 상황에 특히 좋음.
> ```python
> # Focal Loss 코드는 [2.5a 손실 함수 깊게](05a-loss-function.md) 참고
> ```

➡️ [메인으로](README.md)

---

## 불균형 데이터 실전 규칙

- 정확도 단일 지표 사용을 금지합니다.
- 데이터 분할 시 클래스 비율 보존을 기본값으로 둡니다.
- 샘플링/가중치/임계값 조정은 한 번에 하나씩 바꿔 효과를 비교합니다.
- 최종 모델은 소수 클래스 성능과 운영 오탐 비용을 함께 기준으로 선택합니다.



<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **2.8a (보강) 클래스 불균형 다루기** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **2.8a (보강) 클래스 불균형 다루기**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "2.8a (보강) 클래스 불균형 다루기 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **2.8a (보강) 클래스 불균형 다루기**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

2.8a (보강) 클래스 불균형 다루기는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
