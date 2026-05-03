# 2.4 첫 모델 학습: SVM

## 가장 짧은 머신러닝 코드

지금까지의 모든 준비를 거쳤으면, 모델을 학습시키는 건 정말 짧아요. 진짜 짧습니다.

```python
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

이 4줄이 머신러닝 모델의 학습과 예측 전부예요. 진짜로요. **모든 sklearn 모델이 이 패턴**입니다. 외워두시면 평생 갑니다.

---

## SVM이 뭐예요?

**SVM (Support Vector Machine, 서포트 벡터 머신)** 은 1990년대부터 사랑받아온 분류 알고리즘이에요.

핵심 아이디어를 한 문장으로 표현하면 이거예요.

> **두 그룹을 가장 안전한 거리로 가르는 선을 찾자.**

그림으로 보면 이래요.

```
          ●  ●               
       ●     ●                 ← 그룹 A (●)
        ●  ●                  
                              
                              
   ━━━━━━━━━━━━━━━━━━━━━━     ← SVM이 찾은 선
                              
                              
        ×  ×  ×              
       × ×                    ← 그룹 B (×)
         × ×                  
```

그냥 두 그룹을 가르는 선은 무수히 많아요. 그 중에서 SVM은 **양쪽 그룹과의 거리가 가장 먼 선**을 골라요. "안전 거리"를 최대화하는 거죠.

```
         ●  ●                
      ●     ●               
       ●  ●                   ↓
                          여기 선을 그리면
   ━━━━━━━━━━━━━━━━━━━━━━     양쪽이 가장 멀리 떨어짐
                              ↑
       × ×                  
        ×  ×                
         × × ×              
```

이 "안전 거리"를 **마진(margin)** 이라고 부릅니다. SVM은 마진을 최대화하는 알고리즘이에요. 그래서 처음 보는 데이터에도 잘 일반화됩니다.

### kernel이 뭐예요?

위 그림은 직선으로 나누어지는 경우였어요. 그런데 데이터가 항상 직선으로 나뉘는 건 아니잖아요.

```
       ●               ●
   ●       ×       ●
       ×   ×   ×        
   ●       ×       ●
       ●               ●
```

이런 데이터는 **곡선이 필요**해요. SVM은 `kernel`이라는 옵션으로 선의 모양을 바꿀 수 있어요.

| kernel | 모양 | 언제? |
|--------|------|------|
| `'linear'` | 직선 | 데이터가 직선으로 나뉠 때 (간단한 경우) |
| `'rbf'` | 곡선 (사실상 무엇이든) | 거의 모든 경우의 기본값 |
| `'poly'` | 다항식 곡선 | 특정 패턴이 있을 때 |
| `'sigmoid'` | 시그모이드 | 거의 안 씀 |

처음에는 그냥 **`'rbf'`** 부터 시도해 보시면 됩니다. 그게 가장 자주 쓰이고 가장 잘 동작해요.

---

## 첫 모델 학습 — 풀 코드

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 데이터 준비
print("MNIST 다운로드 중... (1~2분)")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(np.int64)

# 빠른 실험을 위해 1000개만
X = X[:1000]
y = y[:1000]

# 2. 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 정규화 (픽셀이라 0~1로)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. 모델 생성
model = SVC(kernel='linear', random_state=42)

# 5. 학습
print("학습 중...")
model.fit(X_train, y_train)
print("학습 완료!")

# 6. 예측
y_pred = model.predict(X_test)

# 7. 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")  # 0.8550 정도
```

이 코드를 Colab에서 그대로 실행해 보세요. 약 1~2분 안에 결과가 나옵니다.

```
MNIST 다운로드 중... (1~2분)
학습 중...
학습 완료!
정확도: 0.8550
```

**85.5% 정확도!** 손글씨 200장 중에 약 171장을 맞혔다는 뜻이에요. 첫 모델치고 나쁘지 않죠?

---

## 코드 한 줄씩 뜯어보기

### `model = SVC(kernel='linear', random_state=42)`

모델을 만들기만 했어요. 아직 학습은 안 했죠. **그냥 모델의 "설정"** 만 정한 거예요. (모델의 "껍데기"라고 보셔도 돼요.)

`SVC` 안의 옵션들:
- `kernel='linear'`: 직선으로 나누기
- `random_state=42`: 학습 결과 재현 가능하게

### `model.fit(X_train, y_train)`

**여기가 진짜 학습**이에요. SVM이 X_train과 y_train을 보고 "어떻게 나눠야 하지?" 하고 안전 거리가 최대인 선을 찾아내요.

데이터가 많으면 시간이 좀 걸려요. 1000개면 몇 초, 60000개면 몇 분~몇십 분.

### `y_pred = model.predict(X_test)`

학습된 모델에 새 데이터(X_test)를 주고 답을 받아요. y_pred는 X_test 각 샘플에 대한 모델의 예측값이에요.

```python
print(y_test[:10])    # 실제 정답
print(y_pred[:10])    # 모델 예측

# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]   ← 다 맞혔네!
```

### `accuracy = accuracy_score(y_test, y_pred)`

정답과 예측을 비교해서 **맞힌 비율**을 계산해요. 이게 정확도(accuracy).

```python
# 직접 계산해도 돼요
correct = (y_test == y_pred).sum()
total = len(y_test)
print(correct / total)
```

`(y_test == y_pred)`는 `True/False` 배열이고, `True`는 1로 더해지니까 정답 수를 셀 수 있어요. (1장의 bool 트릭 기억나시죠?)

---

## predict 말고도 있어요: predict_proba

`predict`는 그냥 카테고리 하나만 답해요. 그런데 가끔 **"얼마나 확신하는지"** 도 알고 싶을 때가 있죠.

```python
# 일반 SVC는 기본적으로 안 됨, probability=True 줘야 함
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test[:5])
print(probs)
```

```
[[0.01 0.02 0.03 ... 0.85 0.04 0.02]   ← 첫 샘플은 7번 클래스를 85% 확신
 [0.78 0.05 0.03 ... 0.01 0.02 0.04]   ← 두 번째는 0번 클래스 78%
 ...]
```

각 행은 한 샘플에 대한 **각 클래스의 확률**이에요. 모두 더하면 1이 되죠.

⚠️ 주의: SVC에서 `probability=True`를 주면 학습이 더 느려져요. 확률이 꼭 필요할 때만 쓰세요. 다른 모델(로지스틱 회귀, 랜덤 포레스트)은 기본적으로 `predict_proba`를 지원합니다.

---

## kernel='rbf'로 바꿔보기

`'linear'` 대신 `'rbf'`를 써 보면 보통 정확도가 더 올라가요.

```python
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RBF 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

```
RBF 정확도: 0.9100
```

91% — linear의 85.5%보다 높아졌어요! 곡선이 직선보다 데이터를 더 잘 따라가기 때문입니다.

---

## 모델은 어디로 가나요? 저장과 불러오기

학습한 모델을 매번 다시 학습시킬 필요는 없어요. 디스크에 저장해 두고 필요할 때 불러올 수 있어요.

```python
import pickle

# 저장
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 불러오기
with open('my_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 바로 예측 가능
loaded_model.predict(X_test[:5])
```

또는 sklearn의 `joblib`을 쓰셔도 돼요. 큰 모델에서는 joblib이 더 빠릅니다.

```python
import joblib
joblib.dump(model, 'my_model.pkl')
loaded_model = joblib.load('my_model.pkl')
```

이건 6장(프로덕션)에서 다시 자세히 다룹니다. 지금은 "저장할 수 있구나" 정도만 알면 충분.

---

## 정리: 첫 모델의 6단계

```python
# 1. 라이브러리 가져오기
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 2. 모델 만들기
model = SVC(kernel='rbf', random_state=42)

# 3. 학습 (가장 오래 걸림)
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 평가
acc = accuracy_score(y_test, y_pred)

# 6. 저장 (선택)
import joblib
joblib.dump(model, 'model.pkl')
```

이 6단계를 외우셨다면, **모든 sklearn 모델을 같은 방식으로 쓸 수 있게 되신 겁니다.**

다음 글에서는 SVC를 RandomForestClassifier, KNeighborsClassifier, LogisticRegression으로 바꾸기만 하면 돼요. 정말로요.

---

## 자주 묻는 질문

> **Q. 학습이 너무 오래 걸려요. 어떡하죠?**
>
> SVM은 데이터가 많아지면 학습이 매우 느려져요. 다음 옵션들을 시도해 보세요.
> 1. 데이터를 더 작게 자르기
> 2. `kernel='linear'`로 바꾸기 (RBF보다 빠름)
> 3. 더 빠른 모델로 바꾸기 (RandomForest, LogisticRegression)
>
> 큰 데이터에서 SVM은 거의 안 써요. 작은 데이터(~10000개)에 적합합니다.

> **Q. 정확도가 너무 낮아요. 어떻게 올리죠?**
>
> 여러 가지 방법이 있어요. 다음 글에서부터 차근차근 보여드릴게요.
> 1. 다른 모델 시도 (다음 글)
> 2. 데이터 더 많이 쓰기
> 3. 하이퍼파라미터 튜닝 (2.7)
> 4. 특성 공학 (feature engineering)
> 5. 딥러닝 (4장)

> **Q. SVM이 어떻게 동작하는지 수학으로 알아야 하나요?**
>
> 솔직히 처음 배우실 땐 몰라도 됩니다. **"안전 거리를 최대로 두는 선을 찾는다"** 정도만 알아도 100% 잘 쓰실 수 있어요. 더 깊이 알고 싶어지셨을 때 그때 보세요.

➡️ 다음: [2.5 평가 지표: 정확도만으론 부족해요](05-평가-지표.md)
