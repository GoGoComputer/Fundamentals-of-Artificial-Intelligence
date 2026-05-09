# 7.6 모델 해석 — "왜 그렇게 예측했는가"

## 들어가며

모델이 90% 정확도로 잘 맞힌다고 합시다. 그럼 끝일까요? 실무에서는 그렇지 않습니다.

- 은행 대출 거절: 고객이 "왜 거절됐냐"고 묻습니다.
- 의료 진단: 의사가 "왜 이 진단인지" 알아야 합니다.
- 채용 시스템: 차별 없는지 검증해야 합니다.
- 모델 디버깅: 잘못 예측한 사례를 보며 데이터 문제를 찾아야 합니다.

EU의 GDPR은 자동 의사결정에 대한 "설명을 받을 권리"를 명시합니다. 한국 개인정보보호법도 비슷한 방향으로 가고 있습니다. 모델 해석은 이제 선택이 아니라 의무에 가깝습니다.

이 장에서는 학습된 모델이 어떤 근거로 예측했는지 들여다보는 도구들을 소개합니다.

---

## 1. 해석의 두 종류 — 전역 vs 지역

| 종류 | 질문 | 예시 |
|------|------|------|
| 전역 (Global) | "이 모델 전체적으로 어떤 특성이 중요한가?" | "신용 점수가 가장 큰 영향을 미친다" |
| 지역 (Local) | "이 한 사람에 대해 왜 이렇게 예측했나?" | "김씨는 부채 비율 때문에 거절됐다" |

두 관점은 보완 관계입니다. 전역 해석으로 모델 전반을 점검하고, 지역 해석으로 개별 사례에 답합니다.

---

## 2. Feature Importance — 기본 도구

### 2.1 트리 기반 모델의 자체 importance

Random Forest, XGBoost, LightGBM 같은 트리 모델은 학습 중 각 특성이 분할에 얼마나 기여했는지 자동으로 계산합니다.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.tail(15).plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
```

### 2.2 단점 — 카디널리티 편향

트리의 자체 importance는 카디널리티(고유값 수)가 많은 특성을 과대평가하는 경향이 있습니다. 사용자 ID 같은 무의미한 특성이 1위로 나올 수 있습니다.

이 함정을 피하려면 다음 방법이 더 안전합니다.

### 2.3 Permutation Importance

특성 한 개의 값을 무작위로 섞고, 모델 성능이 얼마나 떨어지는지 측정합니다. 성능이 크게 떨어질수록 그 특성이 중요했다는 뜻입니다.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test,
                                n_repeats=10, random_state=0)
perm = pd.Series(result.importances_mean, index=X_test.columns)
perm.sort_values(ascending=True).tail(15).plot(kind="barh")
plt.title("Permutation Importance")
```

장점:
- 모델 종류 무관 (선형, 신경망, 트리 모두 가능)
- 카디널리티 편향 없음
- 검증 데이터 기준이라 과적합에 강함

단점:
- 계산이 느림 (특성마다 N번 셔플하며 평가)
- 강한 상관관계가 있는 특성 쌍은 둘 다 낮게 나올 수 있음

---

## 3. SHAP — 게임 이론 기반 해석

### 3.1 핵심 아이디어

SHAP(SHapley Additive exPlanations)은 게임 이론의 **Shapley value** 개념에서 출발합니다. "여러 명이 협력해서 얻은 보상을 어떻게 공정하게 나눌까?"라는 문제의 수학적 해법이죠.

머신러닝에 적용하면 이렇습니다.

> 한 예측값을 만들기 위해 각 특성이 얼마만큼씩 "기여"했는가?

각 특성에 SHAP 값이 부여되고, 모든 SHAP 값을 더하면 (모델 평균 예측 대비) 그 사례의 예측값이 나옵니다.

```
예측값 = 평균 예측 + SHAP(나이) + SHAP(소득) + SHAP(부채) + ...
```

이론적으로 공정하고 일관된 분배라는 게 SHAP의 매력입니다.

### 3.2 사용법

```python
import shap

# Tree 기반 모델은 빠른 TreeExplainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 이진 분류면 shap_values는 [class_0, class_1] 리스트
# class 1 (positive)에 대한 해석을 봅니다
shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
```

### 3.3 전역 해석 — Summary Plot

```python
shap.summary_plot(shap_values_pos, X_test)
```

이 그래프는 각 특성을 행으로 두고, 모든 사례의 SHAP 값을 점으로 흩뿌립니다.
- 가로 위치: SHAP 값 (오른쪽일수록 양성 예측에 기여)
- 색깔: 특성값 (빨강=높음, 파랑=낮음)

해석 예시:
- "income"이 빨강일 때(높을 때) 오른쪽에 분포 → 소득이 높으면 양성 예측 증가
- "debt"가 빨강일 때 왼쪽 → 부채 높으면 음성 예측 증가

### 3.4 지역 해석 — Force Plot

한 사례가 왜 그렇게 예측됐는지 봅니다.

```python
i = 0  # 첫 번째 테스트 샘플
shap.force_plot(
    explainer.expected_value[1],
    shap_values_pos[i],
    X_test.iloc[i],
    matplotlib=True
)
```

평균 예측값에서 출발해 각 특성이 어느 쪽으로 얼마만큼 밀어내는지 시각화합니다.

```
[기본값 0.3] ← debt(0.45) ← age(0.25) | income(0.4) → cards(0.3) → [0.7]
              부채 작용                  소득 작용
```

### 3.5 Dependence Plot — 특성의 효과 곡선

특정 특성의 값에 따라 SHAP 값이 어떻게 변하는지 봅니다.

```python
shap.dependence_plot("income", shap_values_pos, X_test)
```

비선형 관계, 임계값, 상호작용을 발견할 수 있습니다. 예를 들어 소득이 어느 지점까지는 SHAP 값이 0에 가깝다가 특정 임계값 이상에서 급격히 증가하는 패턴 같은 것을 잡습니다.

### 3.6 신경망/일반 모델 — KernelExplainer

```python
# 어떤 모델이든 작동하지만 느림
explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100))
shap_values = explainer.shap_values(X_test.iloc[:50])
```

배경 데이터(`X_train.sample`)를 작게 잡지 않으면 매우 오래 걸립니다.

신경망 전용으로는 `DeepExplainer`(TensorFlow/PyTorch), `GradientExplainer`(역전파 활용)가 있어 더 빠릅니다.

---

## 4. LIME — 지역 선형 근사

LIME(Local Interpretable Model-agnostic Explanations)은 SHAP보다 먼저 나온 방법입니다.

### 4.1 작동 방식

특정 사례 근처에서 모델을 **선형 모델로 근사**합니다. 복잡한 결정 경계도 한 점 근방에서는 직선처럼 볼 수 있다는 발상이죠.

1. 설명할 사례 주변에서 무작위로 샘플 생성
2. 진짜 모델로 그 샘플들 예측
3. 샘플들로 단순한 선형 모델 학습
4. 선형 모델의 계수를 "이 사례에서의 특성 영향력"으로 해석

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=["거절", "승인"],
    mode="classification"
)

i = 0
exp = explainer.explain_instance(
    X_test.iloc[i].values,
    rf.predict_proba,
    num_features=10
)
exp.show_in_notebook()  # 또는 exp.as_pyplot_figure()
```

### 4.2 SHAP vs LIME

| 항목 | SHAP | LIME |
|------|------|------|
| 이론 기반 | Shapley value (수학적으로 공정) | 지역 선형 근사 |
| 일관성 | 보장 | 보장 안 됨 (시드에 따라 다름) |
| 계산 속도 | TreeExplainer는 빠름, 그 외 느림 | 빠름 |
| 시각화 | 풍부 (요약, force, dependence) | 단순 |
| 사용 추천 | 트리 모델, 충분한 시간 | 빠른 프로토타입, 비정형 데이터 |

요즘은 SHAP가 사실상 표준입니다. 단, 속도가 중요하거나 텍스트/이미지 같은 비정형 데이터엔 LIME도 유용합니다.

---

## 5. PDP와 ICE — 특성의 효과 시각화

### 5.1 Partial Dependence Plot (PDP)

"이 특성을 0부터 100까지 바꾸면 평균 예측이 어떻게 변할까?"를 보여줍니다.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=["age", "income", ("age", "income")],  # 단일 + 상호작용
)
plt.show()
```

`("age", "income")`처럼 두 특성을 묶으면 2D 히트맵이 나와 상호작용을 볼 수 있습니다.

### 5.2 ICE (Individual Conditional Expectation)

PDP는 평균이라 개별 차이를 가립니다. ICE는 사례마다 곡선을 그려 다양성을 보여줍니다.

```python
PartialDependenceDisplay.from_estimator(
    rf, X_train, features=["age"], kind="both"  # PDP + ICE 함께
)
```

곡선들이 모두 비슷한 모양이면 모델이 그 특성을 일관되게 사용한 것이고, 곡선이 갈라지면 다른 특성과의 상호작용이 강한 것입니다.

---

## 6. 신경망 해석 — 추가 도구

### 6.1 이미지 — Grad-CAM

CNN이 이미지의 어느 부분을 보고 분류했는지 히트맵으로 보여줍니다.

```python
# pytorch-grad-cam 라이브러리 사용
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
target_layers = [model.layer4[-1]]   # 마지막 합성곱 블록

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=img_tensor)[0]   # 히트맵
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
```

"고양이"로 분류된 이미지에서 모델이 정말 고양이 얼굴을 봤는지, 아니면 배경의 카펫을 단서로 썼는지 한눈에 보입니다. 후자라면 모델이 데이터 셋의 편향을 학습했다는 신호입니다.

### 6.2 텍스트 — Attention 시각화

Transformer 모델은 어텐션 가중치를 직접 시각화할 수 있습니다.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased",
                                   output_attentions=True)

inputs = tokenizer("The cat sat on the mat", return_tensors="pt")
outputs = model(**inputs)
attentions = outputs.attentions   # (layers, B, heads, L, L)

# 마지막 층의 첫 번째 헤드 시각화
import seaborn as sns
attn = attentions[-1][0, 0].detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
```

"it"이 어떤 단어에 주목했는지 같은 패턴을 볼 수 있습니다. 단, 어텐션이 항상 "이유"를 정확히 보여주는 건 아니라는 점은 알아두세요. 최근 연구는 어텐션과 모델 출력의 관계가 생각보다 약할 수 있다고 지적합니다.

---

## 7. 공정성과 편향 점검

해석 도구는 모델의 공정성을 점검하는 데도 쓰입니다.

### 7.1 보호 속성에 대한 의존성

```python
# 성별, 인종 등 보호 속성의 SHAP 값 분포
shap.summary_plot(shap_values_pos[:, X_test.columns.get_loc("gender")],
                  X_test["gender"])
```

보호 속성의 SHAP 분포가 한쪽으로 치우치면, 모델이 그 속성으로 직접 차별하고 있다는 신호입니다.

### 7.2 그룹별 성능 차이

```python
from sklearn.metrics import accuracy_score, recall_score

for group in X_test["gender"].unique():
    mask = X_test["gender"] == group
    pred = rf.predict(X_test[mask])
    print(f"{group}: acc={accuracy_score(y_test[mask], pred):.3f}, "
          f"recall={recall_score(y_test[mask], pred):.3f}")
```

정확도가 비슷해도 그룹별 recall 차이가 크면 공정성 문제일 수 있습니다.

### 7.3 대안 — Counterfactual

"만약 이 사람의 성별만 바꿨다면 예측이 달라졌을까?"를 직접 확인합니다.

```python
test_male = X_test.iloc[[0]].copy()
test_female = test_male.copy()
test_female["gender"] = "F" if test_male["gender"].iloc[0] == "M" else "M"

print("원본:", rf.predict_proba(test_male))
print("성별만 변경:", rf.predict_proba(test_female))
```

차이가 크면 직접적 차별이 있는 것입니다. 보호 속성을 학습 시 제외해도, 우편번호 같은 대리 변수(proxy)를 통해 간접 차별이 생길 수 있어 주의가 필요합니다.

---

## 8. 실전 흐름 — 신용 거절 사례 해석

```python
import shap
import pandas as pd

# 모델은 이미 학습됨 (rf)
explainer = shap.TreeExplainer(rf)

# 거절된 사례 한 명
rejected = X_test[rf.predict(X_test) == 0].iloc[0:1]
print("이 고객의 정보:")
print(rejected.T)

shap_values = explainer.shap_values(rejected)
positive_shap = shap_values[1] if isinstance(shap_values, list) else shap_values

# 영향력 큰 순서로 정리
contrib = pd.Series(positive_shap[0], index=rejected.columns)
contrib = contrib.sort_values()

print("\n거절 사유 (음수일수록 거절 방향):")
print(contrib.head(5))   # 가장 음수인 5개

print("\n승인 방향으로 작용한 요인:")
print(contrib.tail(5))   # 가장 양수인 5개
```

이 결과를 사람이 읽기 좋은 문장으로 바꿉니다.

```python
def explain(contrib, top_n=3):
    negatives = contrib.head(top_n)
    positives = contrib.tail(top_n)
    msg = "거절 결정에 영향을 미친 주요 요인입니다.\n"
    for col, val in negatives.items():
        msg += f"  - {col}: 부정적 영향 (영향력 {val:+.3f})\n"
    msg += "\n승인 방향으로 작용한 요인:\n"
    for col, val in positives.items():
        msg += f"  - {col}: 긍정적 영향 (영향력 {val:+.3f})\n"
    return msg

print(explain(contrib))
```

이 형식이라면 고객 응대, 규제 보고, 내부 검증 모두에 활용 가능합니다.

---

## 9. 해석 도구 사용 시 주의점

### 9.1 상관 ≠ 인과

SHAP가 "소득이 영향력 1위"라고 해도, 그게 "소득을 올리면 승인된다"는 뜻은 아닙니다. 학습 데이터의 패턴을 반영할 뿐, 인과 관계 증명은 별개의 작업입니다.

### 9.2 모델이 다르면 해석도 다름

Random Forest와 XGBoost가 같은 데이터에서 비슷한 정확도를 내도, SHAP 값 분포는 꽤 다를 수 있습니다. 어느 모델의 해석이 "진실"인지 단정하기 어렵습니다.

### 9.3 해석을 너무 믿지 말기

해석 도구도 결국 또 다른 모델/근사입니다. SHAP의 가정(특성 독립성 등)이 깨지면 결과가 왜곡됩니다.

### 9.4 사후 해석 vs 본질적으로 해석 가능한 모델

- **사후 해석**: 블랙박스 모델 학습 후 SHAP 등으로 들여다보기
- **본질적 해석**: 처음부터 해석 가능한 모델 사용 (선형 회귀, 결정 트리, GAM)

리스크가 큰 도메인(의료, 금융 규제)에서는 후자가 더 안전한 선택일 수 있습니다.

---

## 10. 한눈에 정리

| 도구 | 전역/지역 | 모델 종류 | 한 줄 요약 |
|------|----------|-----------|-----------|
| Feature Importance | 전역 | 트리 | 자체 importance, 빠르지만 편향 |
| Permutation Importance | 전역 | 모든 모델 | 셔플 후 성능 저하 측정 |
| SHAP | 둘 다 | 모든 모델 | Shapley value 기반, 표준 도구 |
| LIME | 지역 | 모든 모델 | 근방 선형 근사 |
| PDP/ICE | 전역 | 모든 모델 | 특성 효과 곡선 |
| Grad-CAM | 지역 | CNN | 이미지에서 본 영역 시각화 |
| Attention | 지역 | Transformer | 단어 간 주목 패턴 |

**현장 적용 체크리스트**
- [ ] 전역 importance로 모델 검증 (말도 안 되는 특성이 1위인가?)
- [ ] 지역 SHAP로 잘못 예측한 사례 분석
- [ ] 보호 속성에 대한 의존성 점검
- [ ] 그룹별 성능 차이 확인
- [ ] 사용자에게 보일 형식으로 해석 결과 변환

---

## 더 깊이 가고 싶으시면

- **「Interpretable Machine Learning」** by Christoph Molnar — 무료 온라인 책, 해석 도구의 결정판
- **Causal Inference**: do-calculus, DAG, propensity score — 인과 추론 별도 학습
- **Anchors**: LIME 후속, "이 조건이면 항상 같은 예측" 규칙 추출
- **Counterfactual Explanations**: "최소한의 변경으로 결과를 바꾸려면?"
- **Concept Bottleneck Models**: 사람이 해석 가능한 개념을 거치는 모델
- **fairlearn, aif360**: 공정성 평가/완화 라이브러리

다음 장에서는 지도/비지도와는 다른 패러다임인 **강화학습**을 살펴봅니다. 시행착오로 배우는 AI입니다.
