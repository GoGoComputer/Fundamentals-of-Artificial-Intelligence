# 7.8 AI 윤리와 공정성 — 모델이 옳게 작동한다는 것

## 들어가며

지금까지 우리는 "모델이 잘 작동하는가"를 정확도, F1, AUC 같은 숫자로 평가했습니다. 그런데 현실에서는 또 하나의 질문이 있습니다.

> 이 모델이 **옳게** 작동하는가?

두 질문은 다릅니다.

- 정확도 95% 채용 모델이 여성 지원자를 일관되게 떨어뜨린다면?
- 90% 정확한 의료 진단 모델이 흑인 환자에게서 정확도가 60%로 떨어진다면?
- 사기 탐지 모델이 우편번호로 특정 동네 사람을 더 자주 의심한다면?

전부 실제로 일어났던 일입니다. 모델은 데이터의 패턴을 그대로 학습합니다. 그 데이터에 사회의 편견이 담겨 있다면, 모델은 그 편견을 자동화하고 확대합니다.

이 장에서는 ML 시스템을 만드는 사람이 알아야 할 윤리적 고려사항을 다룹니다.

---

## 1. 실제 사고 사례 — 무엇이 잘못됐나

### 1.1 채용 시스템 (2018, Amazon)

아마존이 10년치 이력서로 학습한 채용 모델이 여성 지원자에게 낮은 점수를 줬습니다. 학습 데이터(과거 합격자) 자체가 남성 위주였기 때문입니다. "여자 체스 동아리"처럼 "여성"이라는 단어가 들어간 이력서를 자동으로 깎았습니다. 결국 프로젝트는 폐기됐습니다.

**교훈**: "성별을 빼고 학습"해도, 동아리 이름·전공·표현 방식 같은 **대리 변수**를 통해 차별이 우회 진입합니다.

### 1.2 재범 예측 (2016, COMPAS)

미국 법원에서 사용된 재범 위험도 모델 COMPAS가 흑인 피고에게 백인 피고보다 두 배 더 자주 "고위험" 판정을 내렸습니다. 정확도는 두 그룹에서 비슷했지만, 위양성률(false positive)이 흑인에게서 훨씬 높았습니다.

**교훈**: "전체 정확도"는 그룹 간 차별을 가립니다. 그룹별로 측정해야 합니다.

### 1.3 의료 영상 (2019)

한 피부암 진단 모델이 흰 피부에서는 잘 작동했지만 어두운 피부에서는 정확도가 급락했습니다. 학습 데이터 대부분이 백인 환자였기 때문입니다. 같은 병이라도 피부색에 따라 양상이 다른데, 그 다양성이 데이터에 부족했습니다.

**교훈**: 데이터의 대표성은 모델 성능과 직결됩니다.

### 1.4 안면 인식

여러 상용 안면 인식 시스템을 평가한 연구(Buolamwini & Gebru, 2018)에서 흑인 여성에 대한 오류율이 백인 남성보다 30%포인트 이상 높게 나왔습니다. 학습 데이터의 인구 통계 편향이 직접 반영된 결과입니다.

---

## 2. 편향(Bias)은 어디서 오는가

ML 시스템에서 편향은 한 곳에서만 들어오지 않습니다. 파이프라인 전 단계에서 들어옵니다.

### 2.1 데이터 수집 단계

- **표본 편향(Sampling bias)**: 특정 집단이 과소/과대 표본됨
- **역사적 편향(Historical bias)**: 과거의 사회적 불평등이 데이터에 새겨짐
- **측정 편향(Measurement bias)**: 측정 방식이 그룹마다 다른 정확도

```
"채용 합격 데이터"가 학습 데이터라면,
과거의 차별적 채용 결과가 곧 모델의 정답이 됨.
```

### 2.2 라벨링 단계

- **라벨러 편향**: 사람의 주관이 라벨에 들어감
- **기준 모호**: "독성 발언" 정의가 라벨러마다 다름
- **언어/문화 편향**: 영어 화자 라벨러의 기준이 모든 언어에 적용됨

### 2.3 모델링 단계

- **목적 함수 편향**: "전체 손실 최소화"는 소수 그룹을 희생할 수 있음
- **알고리즘 편향**: 모델 구조 자체가 특정 패턴에 유리

### 2.4 배포 단계

- **사용 맥락 편향**: 학습한 환경과 다른 환경에서 사용
- **피드백 루프**: 모델 결정이 다음 학습 데이터를 만듦

```
예측 모델 → 자원 배분 결정 → 결과 데이터 → 다음 모델 학습
                                              ↑
                                  모델이 만든 패턴이 재학습됨
```

---

## 3. 공정성(Fairness)의 정의 — 한 가지가 아니다

"공정성"은 단일한 정의가 없습니다. 여러 정의가 있고, 일부는 서로 충돌합니다.

### 3.1 그룹 공정성 지표

이진 분류에서 보호 속성 A(예: 성별)에 대해 정의되는 주요 지표들입니다.

**1) Demographic Parity (인구 통계 평등)**

$$P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1)$$

각 그룹에서 양성 예측 비율이 같아야 한다.

**2) Equal Opportunity (기회 균등)**

$$P(\hat{Y}=1 | Y=1, A=0) = P(\hat{Y}=1 | Y=1, A=1)$$

진짜 양성인 사람들 중에서 양성으로 예측되는 비율(recall)이 그룹 간 같아야 한다.

**3) Equalized Odds (오차율 균등)**

$$P(\hat{Y}=1 | Y=y, A=0) = P(\hat{Y}=1 | Y=y, A=1), \forall y$$

TPR과 FPR이 모두 그룹 간 같아야 한다. 가장 엄격한 조건.

**4) Predictive Parity (예측 평등)**

$$P(Y=1 | \hat{Y}=1, A=0) = P(Y=1 | \hat{Y}=1, A=1)$$

양성 예측의 정확도(precision)가 그룹 간 같아야 한다.

### 3.2 불가능성 정리

수학적으로 증명된 사실: **세 지표(equalized odds, predictive parity, calibration)를 모두 동시에 만족하는 모델은 (특수한 경우를 제외하면) 불가능합니다.**

즉 "어떤 공정성을 우선할 것인가"는 가치 판단의 문제이지, 수학으로 결정할 수 없습니다.

### 3.3 코드로 측정하기

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

def fairness_metrics(y_true, y_pred, group):
    """그룹별 주요 공정성 지표 계산"""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})
    results = {}
    for g in df["group"].unique():
        mask = df["group"] == g
        tn, fp, fn, tp = confusion_matrix(
            df.loc[mask, "y_true"], df.loc[mask, "y_pred"]
        ).ravel()
        results[g] = {
            "selection_rate": (tp + fp) / (tp + fp + tn + fn),
            "tpr_recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "ppv_precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        }
    return pd.DataFrame(results).T

print(fairness_metrics(y_test, y_pred, X_test["gender"]))
```

이 표를 보면 어떤 공정성 기준이 어떻게 깨졌는지 한눈에 보입니다.

---

## 4. 편향 완화 기법

편향이 발견됐을 때 어디에서 개입할지에 따라 세 단계가 있습니다.

### 4.1 전처리 (Pre-processing)

데이터를 수정해 편향을 줄입니다.

- **재가중**: 과소 표본 그룹의 가중치 증가
- **데이터 증강**: 소수 그룹 데이터 합성
- **재라벨링**: 일부 라벨을 공정성 기준에 맞게 조정 (논쟁적)

```python
# 그룹별 균형 잡힌 가중치 계산
from sklearn.utils.class_weight import compute_sample_weight

# 보호 속성과 타깃의 결합으로 가중치
combined = X_train["gender"].astype(str) + "_" + y_train.astype(str)
sample_weight = compute_sample_weight("balanced", combined)

model.fit(X_train, y_train, sample_weight=sample_weight)
```

### 4.2 학습 중 (In-processing)

손실 함수에 공정성 제약을 추가합니다.

```
loss = task_loss + λ * fairness_penalty
```

라이브러리: `fairlearn`, `aif360`이 다양한 알고리즘을 제공합니다.

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression()
mitigator = ExponentiatedGradient(
    base_model, constraints=DemographicParity()
)
mitigator.fit(X_train, y_train, sensitive_features=X_train["gender"])

y_pred = mitigator.predict(X_test)
```

### 4.3 후처리 (Post-processing)

이미 학습된 모델의 출력을 그룹마다 다르게 보정합니다.

```python
from fairlearn.postprocessing import ThresholdOptimizer

postprocessor = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    prefit=True
)
postprocessor.fit(X_train, y_train, sensitive_features=X_train["gender"])

y_pred = postprocessor.predict(X_test, sensitive_features=X_test["gender"])
```

장점: 모델 재학습 불필요
단점: 서로 다른 그룹에 다른 기준 적용 → 법적/윤리적 논쟁 여지

---

## 5. 프라이버시 — 모델이 학습 데이터를 기억한다

### 5.1 멤버십 추론 공격

> 이 데이터가 학습에 사용됐는가?

공격자가 모델의 출력 패턴을 분석해 특정 데이터의 학습 참여 여부를 추측할 수 있습니다. 의료 모델이라면 "이 환자가 학습 데이터에 있다 = 이 환자가 그 병이 있을 확률 높음"으로 이어질 수 있습니다.

### 5.2 모델 역공학

학습 데이터를 모델 출력만 보고 복원하려는 공격입니다. LLM에서 특정 입력에 대해 학습 데이터가 그대로 출력되는 사례가 보고됐습니다.

### 5.3 차분 프라이버시 (Differential Privacy)

수학적으로 정의된 프라이버시 보장입니다. 학습 시 그래디언트에 노이즈를 추가해, 한 사람의 데이터 포함 여부가 결과에 미치는 영향을 제한합니다.

```python
# Opacus 라이브러리 사용 (PyTorch용)
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

비용: 정확도가 일반적으로 떨어집니다. 의료, 금융처럼 프라이버시가 중요한 도메인에서 트레이드오프를 감수합니다.

### 5.4 연합 학습 (Federated Learning)

데이터를 모으지 않고, 각자의 디바이스에서 학습한 모델 업데이트만 중앙 서버로 보냅니다. 안드로이드 키보드의 입력 예측 모델이 대표적 사례입니다.

---

## 6. 환경 영향 — AI도 탄소를 배출한다

### 6.1 학습 에너지

GPT-3 학습에 필요한 전기는 미국 가구 100여 가구가 1년간 쓰는 양과 비슷한 수준으로 추정됐습니다(Patterson et al., 2021). 큰 모델의 학습 한 번이 자동차 한 대의 평생 배기량과 맞먹는다는 보고도 있습니다.

### 6.2 배포 에너지

학습은 1회성이지만 추론은 무한히 반복됩니다. 사용량이 많은 모델은 누적 에너지가 학습 에너지를 훨씬 넘어섭니다.

### 6.3 줄이는 방법

- **모델 효율화**: 양자화, 가지치기, 지식 증류
- **하드웨어 효율 데이터센터** 선택
- **재생 에너지 비율 높은 지역**의 클라우드 사용
- **꼭 큰 모델이 필요한지 재검토**

```python
# CodeCarbon 라이브러리로 학습의 탄소 배출 추적
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# ... 모델 학습 ...

emissions = tracker.stop()
print(f"이번 학습 배출량: {emissions:.4f} kg CO2eq")
```

---

## 7. 책임감 있는 AI 시스템 설계 체크리스트

실무에서 ML 시스템을 만들 때 통과시킬 질문 목록입니다.

### 7.1 문제 정의

- [ ] 이 시스템이 정말 필요한가? ML이 아닌 단순한 규칙으로 풀 수 있는가?
- [ ] 잘못된 예측이 사람에게 미치는 피해는 무엇인가?
- [ ] 거짓 양성과 거짓 음성 중 어느 쪽이 더 위험한가?
- [ ] 시스템에 영향받는 이해관계자는 누구인가?

### 7.2 데이터

- [ ] 데이터의 출처와 수집 과정은 투명한가?
- [ ] 모든 관련 그룹이 충분히 표본되어 있는가?
- [ ] 라벨러의 다양성이 보장되어 있는가?
- [ ] 동의 없이 수집된 개인 정보는 없는가?

### 7.3 모델

- [ ] 그룹별로 성능을 측정했는가?
- [ ] 어떤 공정성 정의를 우선하는지 명시했는가?
- [ ] 해석 가능한 도구로 결정 근거를 설명할 수 있는가?
- [ ] 보호 속성의 대리 변수가 모델 결정을 좌우하지 않는가?

### 7.4 배포

- [ ] 사용자에게 AI 사용 사실이 공개되는가?
- [ ] 자동 결정에 이의를 제기할 수단이 있는가?
- [ ] 사람이 최종 검토하는 단계가 있는가? (high-stakes)
- [ ] 운영 중 성능과 공정성을 모니터링하는가?

### 7.5 거버넌스

- [ ] 모델 카드(Model Card)가 작성되어 있는가?
- [ ] 데이터셋 카드(Datasheet)가 작성되어 있는가?
- [ ] 사고 발생 시 책임 소재가 명확한가?
- [ ] 정기 감사(audit) 계획이 있는가?

---

## 8. 모델 카드와 데이터셋 카드

문서화는 책임감 있는 AI의 기본입니다.

### 8.1 Model Card

Google이 제안한 표준 형식입니다. 모델에 대해 다음을 명시합니다.

- 모델 상세 (구조, 학습 일자, 버전)
- 의도된 용도와 사용자
- 의도되지 않은 용도 (이렇게는 쓰지 마세요)
- 평가 지표 (그룹별 분석 포함)
- 학습 데이터 요약
- 윤리적 고려사항
- 권고사항과 한계

간단한 예시:

```markdown
# Model Card: 신용 평가 모델 v1.2

## 의도된 용도
- 개인 신용카드 신청 자동 심사 보조
- 최종 결정은 사람 심사관이 검토

## 의도되지 않은 용도
- 단독 자동 거절 (사람 검토 없이)
- 보험료 산정, 채용에 사용 금지

## 성능
- AUC: 0.85
- 그룹별 TPR: 남 0.78, 여 0.76, 차이 2%p
- 그룹별 FPR: 남 0.12, 여 0.13

## 한계
- 학습 데이터: 2018-2023, 한국 거주자
- 외국인 등록증 보유자에 대한 성능은 검증되지 않음
- 신용 이력 6개월 미만은 입력에서 자동 거절
```

### 8.2 Datasheet for Datasets

데이터셋에 대한 비슷한 문서입니다.

- 수집 동기, 자금 출처
- 수집 방법과 시점
- 라벨링 절차
- 인구 통계 분포
- 알려진 편향과 한계
- 권장/비권장 사용

`Hugging Face`의 데이터셋과 모델 페이지가 이 형식을 따르고 있어, 좋은 참고 자료가 됩니다.

---

## 9. 법적·규제 흐름

### 9.1 EU AI Act (2024 발효)

세계 최초의 포괄적 AI 규제법입니다. 위험도에 따라 4단계로 분류합니다.

| 위험도 | 예시 | 규제 |
|-------|------|------|
| 금지 | 사회 신용 점수, 무차별 안면 인식 | 사용 금지 |
| 고위험 | 채용, 신용, 의료, 법 집행 | 엄격한 의무 (영향 평가, 투명성, 인간 감독) |
| 제한적 | 챗봇 | 사용자 통지 |
| 최소 | 스팸 필터 | 자율 |

### 9.2 한국 AI 기본법 (2024)

신뢰성 검증, 고영향 AI 사업자 의무, 위험 평가 등을 규정합니다.

### 9.3 GDPR과 자동 의사결정

EU의 일반 데이터 보호 규정은 "전적으로 자동화된 의사결정에 종속되지 않을 권리"와 "설명을 받을 권리"를 명시합니다. 한국 개인정보보호법 개정으로 비슷한 권리가 도입됐습니다.

규제는 빠르게 바뀌고 있습니다. 프로젝트 시작 전 해당 도메인의 최신 규제를 반드시 확인하세요.

---

## 10. 실전 — 공정성 점검 코드 템플릿

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def fairness_audit(model, X_test, y_test, sensitive_col):
    """모델의 공정성을 그룹별로 점검합니다."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"전체 정확도: {(y_pred == y_test).mean():.4f}")
    if y_proba is not None:
        print(f"전체 AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print(f"\n=== {sensitive_col}별 분석 ===")

    rows = []
    for group in X_test[sensitive_col].unique():
        mask = X_test[sensitive_col] == group
        n = mask.sum()
        if n == 0:
            continue

        cm = confusion_matrix(y_test[mask], y_pred[mask])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        row = {
            "group": group,
            "n": n,
            "selection_rate": (tp + fp) / n if n else 0,
            "accuracy": (tp + tn) / n if n else 0,
            "tpr": tp / (tp + fn) if (tp + fn) else 0,
            "fpr": fp / (fp + tn) if (fp + tn) else 0,
            "precision": tp / (tp + fp) if (tp + fp) else 0,
        }
        if y_proba is not None and len(np.unique(y_test[mask])) > 1:
            row["auc"] = roc_auc_score(y_test[mask], y_proba[mask])
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    # 그룹 간 차이 진단
    print(f"\n=== 격차 ===")
    print(f"Selection rate 차이: {df['selection_rate'].max() - df['selection_rate'].min():.3f}")
    print(f"TPR 차이 (Equal Opportunity): {df['tpr'].max() - df['tpr'].min():.3f}")
    print(f"FPR 차이: {df['fpr'].max() - df['fpr'].min():.3f}")
    print(f"Precision 차이: {df['precision'].max() - df['precision'].min():.3f}")

    return df

# 사용
audit_df = fairness_audit(model, X_test, y_test, sensitive_col="gender")
```

이 함수를 모든 모델 평가 단계에 끼워넣는 습관을 들이면, "정확도만 보다가 공정성 사고가 터지는" 일을 막을 수 있습니다.

---

## 11. 한눈에 정리

**편향의 출처**
- 데이터 수집, 라벨링, 모델링, 배포 — 어디서든 들어올 수 있음

**공정성 정의 (서로 충돌 가능)**
- Demographic Parity: 양성 예측 비율 동일
- Equal Opportunity: TPR 동일
- Equalized Odds: TPR + FPR 동일
- Predictive Parity: Precision 동일

**개입 단계**
- Pre-processing: 데이터 수정
- In-processing: 학습 중 제약
- Post-processing: 출력 보정

**필수 점검**
- 그룹별 성능 측정
- 모델 카드/데이터셋 카드 작성
- 사람 검토 단계 보장 (high-stakes)
- 사용자 통지와 이의 제기 수단

---

## 더 깊이 가고 싶으시면

- **「Weapons of Math Destruction」** by Cathy O'Neil — ML 시스템의 사회적 폐해
- **「Fairness and Machine Learning」** by Barocas, Hardt, Narayanan — 무료 온라인 책
- **「Atlas of AI」** by Kate Crawford — AI의 물질적·정치적 비용
- **fairlearn (Microsoft), AIF360 (IBM)** — 공정성 도구 라이브러리
- **Model Cards, Datasheets** — Google·MIT의 표준 문서 형식
- **Partnership on AI, AI Now Institute** — 정책과 거버넌스 연구 기관
- **EU AI Act 원문** — 가장 영향력 있는 AI 규제

---

## 마치며

기술이 빠르게 발전할수록 "이걸 써도 되는가"라는 질문이 더 중요해집니다. 좋은 모델을 만드는 능력만큼이나, **만들지 않을 결정**, **다르게 만들 결정**도 능력입니다.

이 책에서 다룬 모든 도구는 가치 중립적이지 않습니다. 어떤 데이터로 학습할지, 어떤 손실 함수를 쓸지, 어떤 임계값을 둘지 — 모두 사회적 가치 판단을 담고 있습니다.

여러분이 만드는 모델이 누군가의 삶에 영향을 미칠 수 있다는 사실을 잊지 마세요. 그 영향이 좋은 방향이 되도록, 이 장에서 배운 도구들을 활용하시기 바랍니다.
