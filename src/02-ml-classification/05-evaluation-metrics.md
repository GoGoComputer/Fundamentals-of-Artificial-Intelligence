# 2.5 평가 지표: 정확도만으론 부족해요

> **선행 학습:** 평가 지표는 기본 확률 개념을 사용합니다.  
> [👉 확률통계 입문 (5.1 평균과 분산)](../../부록/수학/05-확률통계-입문.md) 을 미리 보면 도움이 되지만, 선택입니다.

---

## 정확도가 99%인데 모델이 쓰레기인 경우

이상한 제목이죠? 한 번 읽어 보세요.

**상황**: 의료 AI 회사에서 일하시는데, "암 환자 탐지 모델"을 만들었어요. 테스트 결과:

> **정확도: 99%**

사장님이 좋아하시면서 "출시하자!" 라고 합니다.

그런데 실제로 보니, **모델은 모든 환자에게 "정상"이라고 답하고 있었어요.** 어떻게 99%가 나왔냐면, 데이터에 진짜 암 환자는 1%밖에 없었기 때문입니다.

```
실제: 정상 99명, 암 환자 1명
모델 예측: 모두 "정상"

맞힌 정상: 99명 / 100명 = 99% 정확도!
하지만 진짜 암 환자는 0명 발견 ❌
```

이 모델은 **사람의 목숨이 걸린 진짜 환자를 모두 놓쳤어요.** 정확도 99%는 의미가 없는 거죠.

**이게 분류 평가에서 가장 중요한 깨달음입니다. "정확도(accuracy)"는 거의 항상 충분하지 않다.**

이 글에서는 정확도 외에 **꼭 알아야 할 4가지 지표**를 다룹니다.

1. **혼동 행렬 (Confusion Matrix)** — 모든 지표의 출발점
2. **정밀도 (Precision)** — "내가 맞다고 한 것 중 진짜 맞는 비율"
3. **재현율 (Recall)** — "진짜 맞는 것 중 내가 맞췄다 한 비율"
4. **F1 점수** — 정밀도와 재현율의 조화

---

## 1. 혼동 행렬 (Confusion Matrix)

이게 모든 지표의 시작점이에요. 표로 봅시다.

이진 분류(정상 vs 암)를 예로 들면:

```
                  실제로:
                  정상      암
        ┌─────────────────────┐
   정상 │  TN      │   FN    │  ← 모델이 "정상"이라고 함
모델이→ │          │          │
        │  (옳음)   │ (놓침!) │
        ├─────────┼─────────┤
    암  │   FP     │   TP    │  ← 모델이 "암"이라고 함
        │ (헛소리)  │  (옳음) │
        └─────────────────────┘
```

각 칸의 약자:
- **TP** (True Positive): 진짜 양성 — 암 환자를 암이라고 맞힘 ✅
- **TN** (True Negative): 진짜 음성 — 정상인을 정상이라고 맞힘 ✅
- **FP** (False Positive): 거짓 양성 — 정상인을 암이라고 잘못 ❌
- **FN** (False Negative): 거짓 음성 — 암 환자를 정상이라고 놓침 ❌❌

🚨 **FN(놓침)은 보통 FP(헛소리)보다 더 위험합니다.** 정상인을 암이라고 잘못 진단하면 추가 검사하면 되지만, 진짜 암 환자를 정상이라고 보내면 사람이 죽어요.

### 코드로 보기

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[99  0]   ← TN, FN
#  [ 1  0]]  ← FP, TP
```

좀 전의 "암 0명 발견" 모델이에요.
- TN = 99, FN = 1 (정상 99명 맞힘, 암 환자 1명 놓침)
- FP = 0, TP = 0 (헛소리 안 함, 진짜 발견도 안 함)

### 보기 좋게 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '암'],
            yticklabels=['정상', '암'])
plt.xlabel('실제')
plt.ylabel('예측')
plt.show()
```

색깔 있는 표가 그려져요. 머신러닝 발표 자료에서 정말 자주 보는 그림입니다.

---

## 2. 정밀도 (Precision)

> **"내가 양성이라고 한 것 중에서, 정말로 양성인 비율"**

```
Precision = TP / (TP + FP)
          = 진짜 맞춘 것 / 내가 양성이라고 답한 것 전부
```

- "암이라고 한 사람들 중 진짜 암 환자 비율"
- 100% precision = "내가 양성이라 한 건 다 진짜 양성"
- 낮은 precision = "헛소리(FP)를 자주 함"

**언제 중요?**
- 헛소리(FP)의 비용이 클 때
- 예: 스팸 필터 (정상 메일을 스팸이라고 막으면 큰일!)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, average='binary')
```

---

## 3. 재현율 (Recall)

> **"진짜 양성인 것 중에서, 내가 양성이라고 답한 비율"**

```
Recall = TP / (TP + FN)
       = 진짜 맞춘 것 / 진짜 양성 전부
```

- "암 환자들 중 내가 발견한 비율"
- 100% recall = "진짜 양성은 다 잡아냄"
- 낮은 recall = "놓침(FN)이 많음"

**언제 중요?**
- 놓침(FN)의 비용이 클 때
- 예: 암 진단 (진짜 환자를 놓치면 큰일!)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, average='binary')
```

---

## 4. Precision vs Recall — 트레이드오프

이 둘은 보통 **반대로 움직입니다.** 한쪽을 올리려고 하면 다른 쪽이 떨어져요.

### 비유

회사 채용을 한다고 해 봅시다.

- **엄격한 면접관**: 진짜 확실한 사람만 합격시킴
  - 합격자들 = 다 진짜 좋은 사람 → **정밀도 높음**
  - 하지만 좋은 사람들도 많이 떨어뜨림 → **재현율 낮음**

- **관대한 면접관**: 살짝만 좋아 보여도 합격
  - 좋은 사람들 거의 다 합격시킴 → **재현율 높음**
  - 하지만 안 좋은 사람도 같이 합격 → **정밀도 낮음**

**둘 다 100%는 사실상 불가능합니다.** 균형을 잡아야 해요.

### 어디에 균형을 두는가?

상황에 따라 다릅니다.

| 상황 | 어느 쪽이 중요? |
|------|---------------|
| 암 진단 | **재현율** (놓치면 사람 죽음) |
| 스팸 필터 | **정밀도** (정상 메일 막으면 곤란) |
| 사기 거래 탐지 | **재현율** (놓치면 돈 잃음, 헛소리는 한 번 확인하면 됨) |
| 채용 추천 | **정밀도** (헛 추천 많으면 신뢰 하락) |
| 의료 검사 (1차 스크리닝) | **재현율** (놓치면 안 됨) |
| 의료 진단 (확진) | **정밀도** (잘못 진단 시 환자 부담) |

**한 줄 요약: 놓치면 큰일이면 recall, 헛소리하면 큰일이면 precision.**

---

## 5. F1 점수: 둘의 조화 평균

"정밀도와 재현율 둘 다 중요한데, 균형을 한 숫자로 보고 싶다."
→ **F1 점수**가 답입니다.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

수식이 복잡해 보이지만, 그냥 **"정밀도와 재현율의 조화 평균"** 이에요.

### 잠깐, 조화 평균이 뭐예요?

평균에는 사실 여러 종류가 있어요. 자주 듣는 "평균"은 **산술 평균** 인데:

```
산술 평균: (a + b) / 2
조화 평균: 2 / (1/a + 1/b) = 2ab / (a + b)
```

조화 평균은 **역수의 평균을 다시 역수** 취한 거예요. 비율이나 속도처럼 "역수가 의미 있는" 수에 자주 써요.

### 왜 보통 평균이 아니라 조화 평균?

만약 Precision=1.0, Recall=0.0 이라고 해 봐요.
- 산술 평균 = (1.0 + 0.0) / 2 = 0.5 ← "그래도 50%네!"
- 조화 평균 (F1) = 2×1×0 / (1+0) = 0 ← "재현율이 0이면 0!"

조화 평균은 **둘 중 어느 하나라도 낮으면 결과도 낮아져요.** 그래서 균형을 더 잘 평가합니다.

> **🧮 수학 보충** — 평균의 종류와 조화 평균이 왜 자연스러운 선택인지는 [부록 수학 1a](../../부록/수학/01a-중학수학-심화.md#1a6-중학-확률통계-다지기) 참고.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average='binary')
```

---

## 6. classification_report — 한 번에 다 보기

`classification_report` 함수는 위의 모든 지표를 한 번에 표로 보여줘요. 정말 자주 씁니다.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support

           0       0.93      0.95      0.94       100
           1       0.91      0.89      0.90       100
           2       0.87      0.90      0.88        95
           ...
           9       0.92      0.88      0.90       102

    accuracy                           0.91      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.91      0.91      1000
```

각 클래스별로 precision, recall, f1을 다 보여줘요. 마지막 줄들은:
- **accuracy**: 전체 정확도
- **macro avg**: 클래스별 지표를 단순 평균
- **weighted avg**: 클래스 크기로 가중평균

다중 분류에서는 보통 **weighted avg**가 가장 정확한 종합 점수예요.

---

## 7. ROC와 AUC

이건 좀 더 깊은 주제예요. 처음 보실 땐 가볍게 봐 두세요.

### ROC 곡선

분류 모델은 **확률**을 출력하잖아요. (`predict_proba`)

```
샘플 1: P(암) = 0.85
샘플 2: P(암) = 0.45
샘플 3: P(암) = 0.10
```

기본적으로 **0.5 이상이면 "암"** 이라고 답해요. 그런데 이 **임계값(threshold)**을 바꾸면 어떻게 될까요?

- 임계값 0.3: 더 많이 "암"이라고 함 → recall ↑, precision ↓
- 임계값 0.7: 더 적게 "암"이라고 함 → precision ↑, recall ↓

이 임계값을 **여러 값으로 바꿔가며 그린 그래프**가 ROC 곡선이에요.

### AUC

ROC 곡선 **아래의 면적(Area Under Curve)** 이 AUC입니다.

```
AUC = 1.0 → 완벽한 분류
AUC = 0.5 → 동전 던지기랑 같음 (쓸모 없음)
AUC = 0.9 이상 → 보통 "좋은 모델"
```

```python
from sklearn.metrics import roc_auc_score

# 이진 분류일 때
probs = model.predict_proba(X_test)[:, 1]   # 양성 클래스의 확률
auc = roc_auc_score(y_test, probs)
print(f"AUC: {auc:.4f}")
```

ROC와 AUC는 **이진 분류에서 가장 자주 쓰이는 종합 지표**입니다. 다중 분류에서도 쓸 수는 있지만 좀 복잡해져요.

---

## 8. 결국 어느 지표를 봐야 하나요?

상황별로 정리해 드릴게요. 이 표만 외워두셔도 충분합니다.

| 상황 | 주요 지표 |
|------|----------|
| 클래스가 균형 잡힌 다중 분류 (MNIST 등) | **Accuracy** + Confusion Matrix |
| 클래스가 불균형한 분류 | **F1** + Recall + Precision |
| 이진 분류, 양쪽 클래스의 비용이 비슷 | **F1** + ROC-AUC |
| 이진 분류, 놓침이 더 비쌈 (의료, 사기 탐지) | **Recall** 우선 |
| 이진 분류, 헛소리가 더 비쌈 (스팸, 추천) | **Precision** 우선 |
| 모델 비교 (어느 게 더 좋은지) | **ROC-AUC** (이진), **F1 weighted** (다중) |

---

## 실전 코드: MNIST에 적용

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

# 모델은 이미 학습돼 있다고 가정
y_pred = model.predict(X_test)

# 1. 정확도
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}\n")

# 2. 분류 리포트 (한 번에 다)
print("[분류 리포트]")
print(classification_report(y_test, y_pred))

# 3. 혼동 행렬
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10))
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('혼동 행렬')
plt.show()
```

혼동 행렬을 보면 **"어느 숫자를 어느 숫자로 헷갈리고 있는지"** 가 한눈에 보여요.

예를 들어 "4를 9로 자주 잘못 본다"거나, "7과 1을 잘 헷갈린다" 같은 패턴이요. 이걸 알면 모델을 어떻게 개선할지 힌트가 보여요.

---

## 정리

```python
# 한 번에 다 보기 (가장 자주 씀)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# 혼동 행렬 (시각화)
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

# 개별 지표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average='weighted')
recall_score(y_test, y_pred, average='weighted')
f1_score(y_test, y_pred, average='weighted')

# 이진 분류용
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

---

## 자주 묻는 질문

> **Q. average='binary'와 'weighted'와 'macro' 차이가 뭐예요?**
>
> - `binary`: 이진 분류에서만. 양성 클래스의 지표만 계산.
> - `macro`: 다중 분류. 클래스별로 계산해서 단순 평균. 작은 클래스도 동등하게 취급.
> - `weighted`: 다중 분류. 클래스별로 계산해서 샘플 수로 가중평균. 큰 클래스에 더 큰 가중치.
>
> 일반적으로 다중 분류면 `weighted`를 씁니다. 클래스 불균형이 매우 심하면 `macro`도 같이 보세요.

> **Q. F1, ROC-AUC, classification_report 다 봐야 하나요?**
>
> 시간 절약 팁: **`classification_report` 한 줄로 거의 끝납니다.** F1 score가 그 안에 있어요. ROC-AUC는 이진 분류에서 추가로 보면 됩니다.

> **Q. 클래스가 너무 불균형하면 어떡해요?**
>
> 여러 방법이 있어요. 다음 글에서는 다루지 않지만, 알아두시면 좋은 키워드들:
> - **샘플링 기법**: SMOTE (소수 클래스 늘리기), Undersampling (다수 클래스 줄이기)
> - **클래스 가중치**: 모델에 `class_weight='balanced'` 옵션
> - **임계값 조정**: 0.5 대신 다른 값으로

➡️ 다음: [2.6 여러 모델 비교하기](06-여러-모델.md)
