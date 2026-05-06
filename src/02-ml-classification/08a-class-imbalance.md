# 2.8a (보강) 클래스 불균형 다루기

> 본문 [2.5 평가 지표](05-평가-지표.md)의 "정확도 99%인데 모델이 쓰레기" 사례를 기억하시나요?
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
> # Focal Loss 코드는 [2.5a 손실 함수 깊게](05a-손실-함수-깊게.md) 참고
> ```

➡️ [메인으로](../README.md)
