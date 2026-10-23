# 3장. 머신러닝 — 회귀

> "내일 기온은 몇 도가 될까?"
> "이 집은 얼마짜리일까?"
> "다음 분기 매출은 얼마쯤 될까?"
>
> 이렇게 **숫자**를 예측하는 게 회귀(Regression)입니다.
> 분류와 짝을 이루는 머신러닝의 다른 한 축이에요.

---

## 이 장의 차례

| | 글 | 무엇을 배우나요? | 시간 |
|---|---|---|---|
| 1 | [회귀란 무엇인가](01-what-is-regression.md) | 분류와 다른 점 | 30분 |
| 2 | [데이터를 만나다 (Boston)](02-meet-data.md) | 집값 예측 데이터 | 30분 |
| 3 | [선형 회귀: 가장 단순하지만 강력한](03-linear-regression.md) | 직선 하나로 예측하기 | 1시간 |
| 4 | [회귀 평가 지표](04-evaluation-metrics.md) | MSE, RMSE, MAE, R² | 30분 |
| 5 | [정규화: Lasso, Ridge, ElasticNet](05-regularization.md) | 과적합 막기 + 특성 선택 | 1시간 |
| 6 | [트리 기반 회귀](06-tree-regression.md) | RF, GBM, XGBoost | 1시간 |
| 7 | [현업 체크리스트](07-production-checklist.md) | 회귀 특유의 함정들 | 30분 |

총 **5~7시간**.

## 실습 파일

| 파일 | 무엇? |
|------|------|
| [실습/01_데이터_탐색.py](practice/01_데이터_탐색.md) | Boston 데이터 탐색 |
| [실습/02_선형회귀.py](practice/02_선형회귀.md) | LinearRegression부터 시작 |
| [실습/03_정규화_비교.py](practice/03_정규화_비교.md) | Lasso/Ridge/ElasticNet |
| [실습/04_트리_회귀.py](practice/04_트리_회귀.md) | RandomForest와 XGBoost |
| [실습/05_종합.py](practice/05_종합.md) | 전체 모델 비교 |

---

## 2장과의 관계

이 장은 [2장 (분류)](../02-ml-classification/README.md)와 **거의 쌍둥이**입니다. 코드 패턴이 똑같아요.

```python
# 2장 (분류)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 3장 (회귀)
from sklearn.svm import SVR    # ← C가 R로 바뀜
model = SVR()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

가장 큰 차이는 두 가지예요.

1. **`y`가 카테고리(0,1,2,...)가 아니라 연속된 숫자(1.5, 3.7, 100.0...)**
2. **평가 지표가 정확도가 아니라 MSE, RMSE 등**

이 두 가지만 익히시면 회귀는 거의 끝입니다.

---

## 2장을 안 보신 분에게

이 장은 [2장 분류](../02-ml-classification/README.md)와 거의 동일한 패턴을 사용해요. 다음 내용은 2장에서 자세히 설명되어 있고 여기서는 짧게만 다룹니다.

- `train_test_split` 사용법
- 정규화 (StandardScaler)
- Pipeline
- GridSearchCV / RandomizedSearchCV

가능하시면 2장을 먼저 보시는 걸 권합니다. 시간이 없다면, 막히는 곳에서 2장으로 점프하셔서 보세요.

---

## 이 장을 이해하려면 알아야 할 수학

이 장의 각 파트를 이해하기 위해 참고하면 좋은 수학 자료입니다.

| 파트 | 필요한 수학 | 위치 |
|------|----------|------|
| **3. 선형 회귀** | 1차 함수, 미분 기초 | [중학수학 복습](../appendix/math/01-middle-school-math.md), [미분 입문](../appendix/math/03-calculus-intro.md) |
| **4. 평가 지표** | 통계 기초 (평균, 분산) | [확률통계 입문](../appendix/math/05-probability-stats-intro.md) |
| **5. 정규화** | 최적화, 가중치 개념 | [미분 입문](../appendix/math/03-calculus-intro.md), [ML에 적용하기](../appendix/math/06-ml-application.md) |

**처음 학습할 때:** 코드와 직관으로 먼저 진행하고, 필요할 때만 수학 부록을 펴세요.  
**심화할 때:** 각 섹션의 수학이 왜 필요한지 이해하려면 위의 자료들을 정독하세요.

---

## 이 장을 끝내면

- ✅ 회귀가 분류와 어떻게 다른지 본인의 언어로 설명할 수 있다
- ✅ 선형 회귀의 직관과 한계를 이해한다
- ✅ MSE, RMSE, MAE, R²의 차이와 사용 상황을 안다
- ✅ Lasso, Ridge, ElasticNet — 정규화의 세 형제를 구분할 수 있다
- ✅ Random Forest 회귀를 실전에 쓸 수 있다
- ✅ 회귀 모델의 결과를 해석해서 비즈니스 의사결정에 활용할 수 있다

---

## 회귀에서 특히 주의할 점

회귀는 분류보다 결과 해석 책임이 더 큽니다. 아래 항목은 꼭 점검하세요.

- RMSE가 낮아도 특정 구간(고가/저가)에서 오차가 편향되는지 잔차 플롯으로 확인하기
- 이상치가 있는 데이터에서 MAE와 RMSE를 함께 보고 지표 해석 왜곡 줄이기
- `R²` 하나만으로 모델 우열을 판단하지 않고 단위가 있는 오차(RMSE/MAE)도 같이 보기
- 선형 모델 계수 해석 전 입력 스케일링/다중공선성 여부를 먼저 확인하기

---

## 실습 정리 체크리스트

다음 질문에 답할 수 있으면 3장의 핵심을 잘 잡으신 상태입니다.

- "분류와 회귀의 가장 큰 차이 두 가지는 무엇인가?"
- "왜 Ridge/Lasso/ElasticNet을 같은 데이터에서 비교해야 하는가?"
- "트리 모델은 왜 스케일링 없이도 잘 동작하는가?"
- "잔차 플롯에서 어떤 모양이 나오면 모델 개선이 필요하다고 보는가?"
- "최종 모델을 저장할 때 메타데이터에 반드시 넣어야 할 항목은 무엇인가?"

자, 시작합시다.

➡️ **[3.1 회귀란 무엇인가](01-what-is-regression.md)**

---

## 추가 학습 가이드

회귀의 핵심은 숫자를 맞히는 것이 아니라 오차를 해석하는 능력입니다.

- RMSE/MAE/R2를 함께 보고 서로 다른 의미를 구분하세요.
- 잔차 플롯으로 체계적 오차 패턴을 반드시 확인하세요.
- 모델 선택 시 해석 가능성과 운영 안정성을 같이 고려하세요.
- 결과 단위(원, 시간, 수량)를 잃지 않고 설명하는 습관을 들이세요.
