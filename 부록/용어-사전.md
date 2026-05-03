# 용어 사전

> 머신러닝/딥러닝에서 자주 만나는 용어 50개를 알파벳 순으로 정리했습니다.
> 한국어 번역도 같이 표시했어요.

---

## A

### Accuracy (정확도)
"맞힌 비율." 분류 모델의 가장 기본적인 평가 지표.
`정확도 = 맞춘 개수 / 전체 개수`

### Activation Function (활성화 함수)
신경망의 각 층에서 비선형성을 만드는 함수. ReLU, Sigmoid, Tanh 등.

### Adam
가장 자주 쓰이는 옵티마이저. 학습률을 자동 조정해줘서 튜닝이 덜 필요.

### AUC (Area Under Curve)
ROC 곡선 아래 면적. 0.5~1.0 사이. 1에 가까울수록 좋은 분류기.

### autograd
PyTorch의 자동 미분 시스템. backward()로 호출.

---

## B

### Backpropagation (역전파)
신경망이 손실의 그래디언트를 계산하는 알고리즘. PyTorch가 자동으로 해줌.

### Batch (배치)
한 번에 모델에 넣는 데이터 묶음. 보통 32, 64, 128개.

### Batch Normalization
각 층의 출력을 정규화해서 학습을 안정화하는 기법.

### Bias (편향)
1) 모델: 선형 결합의 상수항 (`y = wx + b`의 b)
2) 학습: 모델이 진짜 패턴을 못 잡고 단순한 답만 하는 것 (Bias-Variance Tradeoff)
3) 데이터: 데이터가 특정 그룹에 편향됨 (윤리적 문제)

---

## C

### Classification (분류)
입력을 카테고리로 분류하는 문제. 강아지/고양이, 0~9 등.

### CNN (Convolutional Neural Network)
이미지 처리에 특화된 신경망. 이 자료에선 안 다룸.

### CrossEntropyLoss
분류에서 가장 자주 쓰는 손실 함수. PyTorch는 자동으로 softmax 적용.

### Cross-Validation (교차검증)
데이터를 K개로 나눠서 K번 학습/평가해서 평균. 안정적인 평가.

---

## D

### DataLoader
PyTorch의 데이터 배치 처리 도구. 자동 셔플, 병렬 로딩 지원.

### Decision Tree (결정 트리)
스무고개식 분류/회귀 모델. Random Forest의 기본 빌딩 블록.

### Deep Learning (딥러닝)
신경망을 깊게(여러 층) 쌓아서 학습하는 방법. ML의 한 갈래.

### Drift (드리프트)
시간이 지나면서 데이터나 모델 성능이 변화하는 현상.
- Data Drift: 입력 분포 변화
- Concept Drift: 입력-출력 관계 변화

### Dropout
학습 시 일부 뉴런을 무작위로 끄는 정규화 기법. 과적합 방지.

---

## E

### Early Stopping
검증 점수가 안 좋아지면 학습을 일찍 중단하는 기법. 과적합 방지.

### Epoch (에포크)
전체 학습 데이터를 한 번 다 본 것. 보통 10~수백 epoch 학습.

### ElasticNet
Lasso와 Ridge를 합친 정규화 회귀 모델.

---

## F

### F1 Score
정밀도와 재현율의 조화 평균. 분류에서 종합 점수로 자주 사용.

### Feature (특성)
모델의 입력 변수. "독립변수", "x"라고도 함.

### Feature Engineering (특성 엔지니어링)
원시 데이터에서 모델이 잘 학습할 수 있는 특성을 만드는 작업.

### Fine-tuning (파인튜닝)
사전학습된 모델을 본인의 데이터로 추가 학습하는 것.

---

## G

### Gradient (그래디언트, 기울기)
손실 함수의 각 가중치에 대한 미분값. 가중치 업데이트 방향을 알려줌.

### Gradient Boosting
이전 모델의 실수를 보완하는 모델을 순차적으로 쌓는 앙상블. XGBoost, LightGBM이 대표.

### GPU
그래픽 처리 장치. 행렬 연산이 빨라 신경망 학습에 필수.

---

## H

### Hyperparameter (하이퍼파라미터)
학습 전에 사람이 정하는 값. 학습률, 트리 개수, 층 수 등. **파라미터**(모델이 학습하는 값)와 다름.

---

## I

### Inference (추론)
학습된 모델로 예측하는 것. `model.predict()` 호출하는 단계.

---

## L

### Label (라벨)
데이터의 정답. 분류에서는 카테고리, 회귀에서는 숫자.

### Lasso
L1 정규화 회귀. 일부 가중치를 0으로 만들어 자동 특성 선택.

### Learning Rate (학습률)
가중치를 한 번에 얼마나 바꿀지. 가장 중요한 하이퍼파라미터. 보통 0.001~0.01.

### LLM (Large Language Model)
거대 언어 모델. GPT, Claude, Gemini 등.

### Loss Function (손실 함수)
모델이 얼마나 틀렸는지를 점수로. MSELoss, CrossEntropyLoss 등.

---

## M

### MAE (Mean Absolute Error)
평균 절대 오차. 회귀 평가 지표.

### Metric (메트릭)
평가 지표. accuracy, F1, RMSE 등.

### Mini-batch
한 번에 처리하는 작은 데이터 묶음 (= Batch).

### MLP (Multi-Layer Perceptron)
가장 기본 신경망 구조. 여러 Dense 층을 쌓은 것.

### MLOps
ML + DevOps. ML 시스템의 자동화와 운영.

### MNIST
손글씨 숫자 데이터셋. ML의 "Hello World".

### Model
학습된 ML 알고리즘. 입력을 받아 예측을 내놓는 함수.

### MSE (Mean Squared Error)
평균 제곱 오차. 가장 기본적인 회귀 손실/지표.

---

## N

### Neural Network (신경망)
여러 층으로 구성된 ML 모델. 딥러닝의 기본.

### NumPy
파이썬의 숫자 배열 라이브러리. ML의 기본 도구.

---

## O

### Optimizer (옵티마이저)
가중치를 업데이트하는 알고리즘. SGD, Adam, AdamW 등.

### Overfitting (과적합)
학습 데이터를 너무 잘 외워서 새 데이터에 못하는 현상.

---

## P

### Pandas
파이썬의 표 데이터 라이브러리. CSV, Excel 같은 데이터 다룸.

### Parameter (파라미터)
모델이 학습하는 값 (가중치 등). **하이퍼파라미터**와 다름.

### Pipeline
여러 처리 단계를 묶은 것. 보통 전처리 + 모델.

### Precision (정밀도)
모델이 양성이라고 한 것 중 진짜 양성 비율.

### Prediction (예측)
모델의 출력. `model.predict(X)`의 결과.

### PyTorch
딥러닝 라이브러리. Meta가 만들고 가장 인기.

---

## R

### Random Forest
여러 결정 트리의 앙상블. 매우 자주 쓰이는 모델.

### Recall (재현율)
진짜 양성 중 모델이 잡은 비율. 1에 가까울수록 안 놓침.

### Regression (회귀)
연속된 숫자를 예측하는 문제. 집값, 매출 등.

### Regularization (정규화 / 규제)
모델이 과적합되지 않게 페널티를 주는 기법. L1, L2, Dropout 등.

### ReLU (Rectified Linear Unit)
가장 자주 쓰는 활성화 함수. `f(x) = max(0, x)`.

### Ridge
L2 정규화 회귀. 가중치를 작게 만들지만 0으로는 안 함.

### RMSE (Root Mean Squared Error)
MSE의 제곱근. 회귀의 가장 표준 지표.

### ROC Curve
이진 분류기의 성능 곡선. 임계값을 바꿔가며 그림.

---

## S

### scikit-learn (sklearn)
파이썬의 ML 라이브러리. 거의 모든 전통 ML 알고리즘 포함.

### SGD (Stochastic Gradient Descent)
가장 기본 옵티마이저. 한 번에 한 배치만 보고 업데이트.

### Sigmoid
값을 0~1 사이로 만드는 활성화 함수. 이진 분류 출력에.

### Softmax
여러 값을 합 1인 확률로 바꿔주는 함수. 다중 분류 출력에.

### SVM (Support Vector Machine)
"안전 거리를 최대로 두는 선" 분류 알고리즘.

---

## T

### Tensor (텐서)
다차원 배열. PyTorch의 기본 단위. NumPy 배열과 비슷.

### Test Set (평가 셋)
모델 학습에 절대 안 쓰고 최종 평가에만 사용하는 데이터.

### TorchScript
PyTorch 모델을 다른 환경(C++, 모바일)에서 쓸 수 있게 변환.

### Train Set (학습 셋)
모델 학습에 쓰는 데이터.

### Transfer Learning (전이 학습)
다른 데이터로 학습된 모델을 가져와서 본인 데이터에 적용.

### Transformer
현대 AI의 핵심 구조. ChatGPT, BERT 등.

---

## U

### Underfitting (과소적합)
모델이 너무 단순해서 학습 데이터도 잘 못 맞추는 현상.

---

## V

### Validation Set (검증 셋)
학습 중에 모델 성능을 평가하는 데이터. 하이퍼파라미터 튜닝에 사용.

---

## W

### Weight (가중치)
신경망 / 선형 모델의 학습되는 값. `y = wx + b`의 w.

### Weight Decay
L2 정규화. 가중치가 너무 커지지 못하게 페널티.

---

## X

### XGBoost
가장 강력한 트리 기반 모델. 캐글에서 자주 우승.

---

## Y

### y_pred
모델의 예측값. y_true와 비교.

### y_true
실제 정답. y_test, target이라고도 함.

---

## Z

### Zero-shot
학습 안 한 새 작업을 모델이 풀어내는 능력. LLM의 핵심 능력 중 하나.

---

## 자주 헷갈리는 짝

### Parameter vs Hyperparameter
- **Parameter**: 모델이 학습하는 값 (가중치 w)
- **Hyperparameter**: 사람이 정하는 값 (학습률, 트리 개수)

### Train vs Test vs Validation
- **Train**: 모델 학습용 데이터
- **Validation**: 학습 중 성능 확인용
- **Test**: 최종 평가용. 절대 학습에 안 씀.

### Loss vs Metric
- **Loss**: 학습 중 줄여야 할 값. 미분 가능해야 함. (MSE)
- **Metric**: 사람이 보는 평가 지표. (RMSE, R²)

### Accuracy vs Precision vs Recall
- **Accuracy**: 전체 중 맞춘 비율
- **Precision**: 양성이라 한 것 중 진짜 양성
- **Recall**: 진짜 양성 중 잡아낸 것

### Bias vs Variance
- **High Bias**: 모델이 너무 단순함 (과소적합)
- **High Variance**: 모델이 너무 복잡함 (과적합)

### Stochastic vs Batch vs Mini-batch
- **Stochastic GD**: 샘플 1개씩
- **Batch GD**: 전체 데이터 한 번에
- **Mini-batch GD**: 32~256개씩 (실제로 가장 많이 씀)
