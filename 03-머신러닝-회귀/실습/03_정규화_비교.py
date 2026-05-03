"""3장 5절 실습: 정규화 (Lasso, Ridge, ElasticNet) 비교

같은 데이터로 4가지 회귀를 학습시켜 가중치와 성능을 비교합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV,
)
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 데이터 준비
# ============================================================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = pd.DataFrame(data, columns=feature_names)
y = pd.Series(target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4가지 모델 학습
# ============================================================
models = {
    "Linear":     LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),
    "Lasso":      Lasso(alpha=0.5),    # alpha 좀 크게 해서 효과 보기
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5),
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    weights = pipe.named_steps['model'].coef_
    n_zero = (np.abs(weights) < 1e-10).sum()

    results[name] = {
        'rmse': rmse,
        'r2': r2,
        'weights': weights,
        'n_zero': n_zero,
    }

    print(f"[{name}]")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  0이 된 가중치: {n_zero}개\n")


# ============================================================
# 가중치 비교 시각화
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(feature_names))
width = 0.2

colors = ['lightgray', 'steelblue', 'salmon', 'mediumseagreen']

for i, (name, r) in enumerate(results.items()):
    offset = (i - 1.5) * width
    ax.bar(x_pos + offset, r['weights'], width, label=name, color=colors[i])

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('가중치 (정규화 후)')
ax.set_title('4가지 회귀 모델의 가중치 비교 (Lasso는 일부가 0)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('regularization_weights.png', dpi=80)
plt.show()
print("저장: regularization_weights.png")


# ============================================================
# 성능 비교
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
rmses = [results[n]['rmse'] for n in names]
r2s = [results[n]['r2'] for n in names]

axes[0].bar(names, rmses, color=colors)
axes[0].set_ylabel('RMSE (낮을수록 좋음)')
axes[0].set_title('RMSE 비교')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(rmses):
    axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(names, r2s, color=colors)
axes[1].set_ylabel('R² (높을수록 좋음)')
axes[1].set_title('R² 비교')
axes[1].set_ylim(0, 1)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(r2s):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('regularization_performance.png', dpi=80)
plt.show()


# ============================================================
# 자동으로 알파 찾기 (CV)
# ============================================================
print("\n" + "=" * 50)
print("CV로 최적 alpha 찾기")
print("=" * 50)

# RidgeCV
rcv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
pipe_rcv = Pipeline([('scaler', StandardScaler()), ('model', rcv)])
pipe_rcv.fit(X_train, y_train)
print(f"\nRidge 최적 alpha: {pipe_rcv.named_steps['model'].alpha_}")
y_pred = pipe_rcv.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# LassoCV
lcv = LassoCV(alphas=[0.001, 0.01, 0.1, 0.5, 1, 5, 10], cv=5, max_iter=10000)
pipe_lcv = Pipeline([('scaler', StandardScaler()), ('model', lcv)])
pipe_lcv.fit(X_train, y_train)
print(f"\nLasso 최적 alpha: {pipe_lcv.named_steps['model'].alpha_:.4f}")
y_pred = pipe_lcv.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


# ============================================================
# alpha에 따른 가중치 변화 (Ridge)
# ============================================================
print("\n[alpha를 키우면 Ridge 가중치들이 어떻게 변할까?]")

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
weights_history = []

for a in alphas:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=a)),
    ])
    pipe.fit(X_train, y_train)
    weights_history.append(pipe.named_steps['model'].coef_)

weights_history = np.array(weights_history)

plt.figure(figsize=(12, 6))
for i, name in enumerate(feature_names):
    plt.plot(alphas, weights_history[:, i], 'o-', label=name)

plt.xscale('log')
plt.xlabel('alpha (정규화 강도)')
plt.ylabel('가중치')
plt.title('alpha를 키우면 가중치가 작아짐 (Ridge)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_path.png', dpi=80)
plt.show()
print("저장: alpha_path.png")
print("\n→ alpha가 클수록 가중치들이 0 쪽으로 모입니다 (정규화의 효과)")


print("\n정규화 비교 끝!")
