"""2장 3~5절 실습: 첫 분류기 만들기

데이터 준비 → 모델 학습 → 평가의 한 사이클을 완전히 돌려봅니다.
이 패턴이 머신러닝의 80%입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# 1단계에서 모델 학습에 필요한 입력 X와 정답 y를 같은 인덱스로 맞춰 준비합니다.
print("[1] 데이터 준비")
print("-" * 40)

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data
y = mnist.target.astype(np.int64)

# 전체 7만 장 대신 2000장만 사용해 학습 시간을 줄이고 실습 반복 속도를 높입니다.
X = X[:2000]
y = y[:2000]

print(f"전체 데이터: X={X.shape}, y={y.shape}")


# train/test 분할로 학습 데이터와 처음 보는 평가 데이터를 분리합니다.
print("\n[2] train/test 분할")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    # stratify를 넣으면 각 숫자 클래스 비율이 train/test에 비슷하게 유지됩니다.
    stratify=y,
)

print(f"학습용: X={X_train.shape}, y={y_train.shape}")
print(f"평가용: X={X_test.shape}, y={y_test.shape}")


# 픽셀값 0~255를 0~1로 바꿔 SVM 학습의 수치 안정성을 높입니다.
print("\n[3] 정규화 (픽셀 0~255 → 0~1)")
print("-" * 40)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"X_train 범위: {X_train.min():.2f} ~ {X_train.max():.2f}")


# 모델 생성과 fit 단계가 머신러닝 학습의 핵심이며 대부분의 sklearn 모델이 같은 패턴을 따릅니다.
print("\n[4] 모델 학습 (SVM with RBF kernel)")
print("-" * 40)

import time
model = SVC(kernel='rbf', random_state=42)

start = time.time()
model.fit(X_train, y_train)
duration = time.time() - start

print(f"학습 시간: {duration:.2f}초")


# train/test 정확도를 같이 보면 과적합 여부를 간단히 가늠할 수 있습니다.
print("\n[5] 예측")
print("-" * 40)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"훈련 정확도: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"평가 정확도: {accuracy_score(y_test, y_test_pred):.4f}")


# classification_report는 클래스별 precision/recall/F1까지 보여줘 정확도의 빈틈을 메워줍니다.
print("\n[6] 분류 리포트")
print("-" * 40)
print(classification_report(y_test, y_test_pred))


# 혼동 행렬은 어떤 숫자쌍을 특히 자주 헷갈리는지 찾을 때 가장 유용합니다.
print("\n[7] 혼동 행렬")
print("-" * 40)

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(10), yticklabels=range(10),
)
plt.xlabel('예측한 숫자')
plt.ylabel('실제 숫자')
plt.title('MNIST 혼동 행렬')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=80)
plt.show()
print("저장: confusion_matrix.png")


# 오분류 샘플을 직접 보면 데이터 난이도와 모델 약점을 동시에 파악할 수 있습니다.
print("\n[8] 잘못 분류된 샘플들")
print("-" * 40)

wrong_idx = np.where(y_test != y_test_pred)[0]
print(f"틀린 샘플 수: {len(wrong_idx)}")

# 처음 10개만 시각화
n_show = min(10, len(wrong_idx))
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    if i >= n_show:
        ax.axis('off')
        continue

    idx = wrong_idx[i]
    img = X_test[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'정답:{y_test[idx]} 예측:{y_test_pred[idx]}', fontsize=10)
    ax.axis('off')

plt.suptitle('잘못 분류된 샘플들', fontsize=14)
plt.tight_layout()
plt.savefig('wrong_predictions.png', dpi=80)
plt.show()
print("저장: wrong_predictions.png")
print("→ 사람이 봐도 헷갈리는 글씨들이 많을 거예요")


# 마지막에 모델 파일을 저장해 두면 다음 실습/서비스 코드에서 재학습 없이 재사용할 수 있습니다.
print("\n[9] 모델 저장")
print("-" * 40)

import joblib
joblib.dump(model, 'svm_mnist.pkl')
print("저장: svm_mnist.pkl")
print("→ joblib.load('svm_mnist.pkl') 으로 다시 불러올 수 있어요")


print("\n" + "=" * 40)
print("첫 분류기 완성!")
print("=" * 40)
