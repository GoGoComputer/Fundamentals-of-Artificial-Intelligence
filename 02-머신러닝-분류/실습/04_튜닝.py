"""2장 7절 실습: RandomizedSearchCV로 하이퍼파라미터 튜닝

기본 옵션 vs 튜닝 후를 직접 비교해 보세요.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint


# ============================================================
# 데이터 준비
# ============================================================
print("[1] 데이터 준비")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data[:3000] / 255.0
y = mnist.target[:3000].astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# 베이스라인 (기본 옵션)
# ============================================================
print("\n[2] 베이스라인 (기본 옵션)")
print("-" * 40)

baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"기본 옵션 정확도: {baseline_acc:.4f}")


# ============================================================
# 튜닝
# ============================================================
print("\n[3] 하이퍼파라미터 탐색 (RandomizedSearchCV)")
print("-" * 40)
print("30가지 조합 × 5-fold CV = 150번 학습")
print("몇 분 걸려요...\n")

# 탐색 공간
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)

print(f"\n[탐색 결과]")
print(f"교차검증 정확도: {search.best_score_:.4f}")
print(f"최고 조합:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")


# ============================================================
# 튜닝된 모델 평가
# ============================================================
print("\n[4] 튜닝된 모델로 test 평가")
print("-" * 40)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
tuned_acc = accuracy_score(y_test, y_pred)

print(f"튜닝 후 정확도: {tuned_acc:.4f}")
print(f"개선: {tuned_acc - baseline_acc:+.4f}")

print("\n분류 리포트:")
print(classification_report(y_test, y_pred))


# ============================================================
# 비교 시각화
# ============================================================
labels = ['베이스라인', '튜닝 후']
accs = [baseline_acc, tuned_acc]
colors = ['lightblue', 'mediumseagreen']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, accs, color=colors, edgecolor='black')
plt.ylabel('정확도')
plt.title('하이퍼파라미터 튜닝 효과')
plt.ylim(min(accs) - 0.02, max(accs) + 0.02)

for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
        f'{acc:.4f}', ha='center', fontsize=12, fontweight='bold'
    )

# 차이 표시
diff = tuned_acc - baseline_acc
plt.text(
    0.5, max(accs) + 0.005,
    f'개선: {diff:+.4f} ({diff*100:+.2f}%)',
    transform=plt.gca().transAxes, ha='center',
    fontsize=11, color='red',
)

plt.tight_layout()
plt.savefig('tuning_effect.png', dpi=80)
plt.show()
print("\n저장: tuning_effect.png")


# ============================================================
# 탐색 과정 분석
# ============================================================
print("\n[5] 탐색 과정 분석")
print("-" * 40)

import pandas as pd
results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)

print("상위 5개 조합:")
for i, (_, row) in enumerate(results_df.head().iterrows()):
    print(f"\n  #{i+1}: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
    for k, v in row['params'].items():
        print(f"     {k}: {v}")


# 탐색 점수 분포
plt.figure(figsize=(10, 5))
plt.hist(results_df['mean_test_score'], bins=20, color='steelblue', edgecolor='black')
plt.axvline(baseline_acc, color='blue', linestyle='--', label=f'베이스라인 {baseline_acc:.4f}')
plt.axvline(search.best_score_, color='red', linestyle='--', label=f'최고 CV {search.best_score_:.4f}')
plt.xlabel('교차검증 정확도')
plt.ylabel('빈도')
plt.title('30가지 조합의 점수 분포')
plt.legend()
plt.tight_layout()
plt.savefig('search_distribution.png', dpi=80)
plt.show()
print("저장: search_distribution.png")


print("""
================================================================
배운 점:
- 30가지 조합 모두가 다 좋은 건 아닙니다 (분포가 넓어요)
- 그래서 한 번에 좋은 조합을 잘 못 찾고, 여러 번 시도하는 거예요
- 이런 자동화 도구가 없으면 사람이 손으로 못 합니다
- 데이터가 더 큰 경우엔 Optuna 같은 더 똑똑한 도구도 고려하세요
================================================================
""")
