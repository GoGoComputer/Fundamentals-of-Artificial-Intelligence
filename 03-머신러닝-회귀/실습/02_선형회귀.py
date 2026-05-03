"""3장 3~4절 실습: 선형 회귀와 평가 지표

가장 단순한 회귀로 시작해서, 지표 4종을 한꺼번에 봅니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# 1. 데이터 준비
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
# 2. Pipeline (정규화 + 모델)
# ============================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression()),
])

pipe.fit(X_train, y_train)


# ============================================================
# 3. 예측
# ============================================================
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)


# ============================================================
# 4. 평가 (4가지 지표 다)
# ============================================================
def evaluate(y_true, y_pred, name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[{name}]")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}    (단위: $1000)")
    print(f"  MAE:  {mae:.4f}    (단위: $1000)")
    print(f"  R²:   {r2:.4f}    (1.0이 완벽)")
    print()

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


train_metrics = evaluate(y_train, y_train_pred, "Train")
test_metrics = evaluate(y_test, y_test_pred, "Test")


# 과적합 체크
gap = train_metrics['r2'] - test_metrics['r2']
print(f"R² 차이 (Train - Test): {gap:.4f}")
if gap > 0.1:
    print("→ 과적합 의심 (정규화 필요?)")
elif gap < 0.05:
    print("→ 과적합 없음 (좋음)")


# ============================================================
# 5. 가중치 분석
# ============================================================
weights = pd.DataFrame({
    'feature': X.columns,
    'weight': pipe.named_steps['lr'].coef_,
})
weights['abs_weight'] = weights['weight'].abs()
weights = weights.sort_values('abs_weight', ascending=False)

print("\n[가중치 (정규화 후, 절대값 큰 순)]")
print(weights[['feature', 'weight']].to_string(index=False))


# 가중치 시각화
plt.figure(figsize=(12, 5))
sorted_weights = weights.sort_values('weight')
colors = ['salmon' if w < 0 else 'steelblue' for w in sorted_weights['weight']]
plt.barh(sorted_weights['feature'], sorted_weights['weight'], color=colors)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('가중치')
plt.title('Linear Regression 가중치 (음수=가격 ↓, 양수=가격 ↑)')
plt.tight_layout()
plt.savefig('linear_weights.png', dpi=80)
plt.show()
print("저장: linear_weights.png")


# ============================================================
# 6. 예측 vs 실제
# ============================================================
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='완벽 (y=x)')
plt.xlabel('실제 집값')
plt.ylabel('예측 집값')
plt.title(f'예측 vs 실제\nRMSE={test_metrics["rmse"]:.2f}, R²={test_metrics["r2"]:.4f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('linear_pred_vs_actual.png', dpi=80)
plt.show()
print("저장: linear_pred_vs_actual.png")


# ============================================================
# 7. 잔차 플롯 (모델 진단의 핵심!)
# ============================================================
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 잔차 vs 예측값
axes[0].scatter(y_test_pred, residuals, alpha=0.5, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('예측값')
axes[0].set_ylabel('잔차 (실제 - 예측)')
axes[0].set_title('잔차 플롯')
axes[0].grid(alpha=0.3)

# 잔차 분포
axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('잔차')
axes[1].set_ylabel('빈도')
axes[1].set_title('잔차 분포 (정규분포에 가까운가?)')

plt.tight_layout()
plt.savefig('linear_residuals.png', dpi=80)
plt.show()
print("저장: linear_residuals.png")
print("\n→ 잔차에 패턴이 보이면 비선형 관계 있음")
print("→ 잔차 분포가 비대칭이면 log 변환 고려")


print("\n선형 회귀 끝! 이제 정규화로 더 좋게 만들어 봅시다.")
