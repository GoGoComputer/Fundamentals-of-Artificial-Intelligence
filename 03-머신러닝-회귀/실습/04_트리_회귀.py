"""3장 6절 실습: 트리 기반 회귀 모델 비교

선형 모델과 트리 모델의 성능 차이를 직접 확인합니다.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score

# XGBoost (있으면 추가)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("(XGBoost 없음. pip install xgboost로 설치하시면 더 비교 가능)")


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
# 모델 정의
# ============================================================
# Ridge는 정규화 필요, 트리는 안 필요
# Pipeline으로 처리
models = {
    "Ridge (선형)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0)),
    ]),
    "Decision Tree (단일)": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42,
    ),
}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0,
    )


# ============================================================
# 학습 + 평가
# ============================================================
results = []

for name, model in models.items():
    print(f"[{name}] 학습 중...")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 훈련 점수도 (과적합 체크)
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)

    results.append({
        'name': name,
        'rmse': rmse,
        'r2': r2,
        'train_r2': train_r2,
        'gap': train_r2 - r2,
        'time': train_time,
        'model': model,
    })

    print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f} (Train R²: {train_r2:.4f}), {train_time:.2f}s\n")


# ============================================================
# 결과 표
# ============================================================
results.sort(key=lambda r: r['rmse'])

print("=" * 75)
print(f"{'모델':<25} {'RMSE':>8} {'Test R²':>10} {'Train R²':>10} {'Gap':>8} {'시간(s)':>8}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<25} {r['rmse']:>8.4f} {r['r2']:>10.4f} "
          f"{r['train_r2']:>10.4f} {r['gap']:>8.4f} {r['time']:>8.2f}")
print("=" * 75)

print("\nGap이 크면 과적합. 단일 트리는 거의 항상 과적합돼요.")


# ============================================================
# 시각화: RMSE
# ============================================================
plt.figure(figsize=(10, 5))
names = [r['name'] for r in results]
rmses = [r['rmse'] for r in results]

bars = plt.barh(names, rmses, color=['lightgray' if 'Ridge' in n else
                                     'salmon' if 'Tree' in n else
                                     'steelblue' for n in names])
plt.xlabel('RMSE (낮을수록 좋음)')
plt.title('회귀 모델 비교 (Boston Housing)')
for i, v in enumerate(rmses):
    plt.text(v + 0.05, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('tree_models_compare.png', dpi=80)
plt.show()
print("저장: tree_models_compare.png")


# ============================================================
# 특성 중요도 (Random Forest)
# ============================================================
rf = next(r['model'] for r in results if r['name'] == 'Random Forest')
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values()

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('중요도')
plt.title('Random Forest 특성 중요도')
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=80)
plt.show()
print("저장: rf_importance.png")


# ============================================================
# 예측 vs 실제 (베스트 모델)
# ============================================================
best = results[0]
y_pred = best['model'].predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', label=f'{best["name"]}')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='완벽 (y=x)')
plt.xlabel('실제 집값')
plt.ylabel('예측 집값')
plt.title(f'{best["name"]} — 예측 vs 실제\nRMSE={best["rmse"]:.2f}, R²={best["r2"]:.4f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('best_pred_vs_actual.png', dpi=80)
plt.show()
print("저장: best_pred_vs_actual.png")


print(f"\n베스트 모델: {best['name']} (RMSE: {best['rmse']:.4f})")
print("\n트리 비교 끝!")
