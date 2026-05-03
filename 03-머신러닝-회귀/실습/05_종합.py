"""3장 종합 실습: 모든 회귀 모델 비교 + 튜닝

실제 회사에서 회귀 문제를 만났을 때 어떻게 접근하는지를
한 파일에 다 담았습니다. 길지만 따라오시면 큰 그림이 잡힙니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy.stats import randint, uniform


# ============================================================
# 1단계: 데이터 준비
# ============================================================
print("=" * 60)
print("[1단계] 데이터 준비")
print("=" * 60)

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
print(f"학습: {X_train.shape}, 평가: {X_test.shape}")


# ============================================================
# 2단계: 베이스라인 모델들 빠르게 비교
# ============================================================
print("\n" + "=" * 60)
print("[2단계] 베이스라인 모델 비교 (모두 기본 옵션)")
print("=" * 60)

baseline_models = {
    "Linear":            Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
    "Ridge":             Pipeline([('s', StandardScaler()), ('m', Ridge())]),
    "Lasso":             Pipeline([('s', StandardScaler()), ('m', Lasso(alpha=0.1))]),
    "ElasticNet":        Pipeline([('s', StandardScaler()), ('m', ElasticNet(alpha=0.1))]),
    "SVR (RBF)":         Pipeline([('s', StandardScaler()), ('m', SVR(kernel='rbf'))]),
    "Random Forest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

baseline_results = {}
for name, model in baseline_models.items():
    # 5-fold CV로 안정적 측정
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
    )
    rmse_cv = np.sqrt(-scores.mean())

    # test도 평가
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    baseline_results[name] = {'cv_rmse': rmse_cv, 'test_rmse': rmse_test}
    print(f"  {name:25s}  CV RMSE: {rmse_cv:.3f}  Test RMSE: {rmse_test:.3f}")


# 베스트 베이스라인
best_baseline = min(baseline_results.items(), key=lambda x: x[1]['cv_rmse'])
print(f"\n베스트 베이스라인: {best_baseline[0]} (CV RMSE: {best_baseline[1]['cv_rmse']:.3f})")


# ============================================================
# 3단계: 베스트 모델 튜닝 (Random Forest 가정)
# ============================================================
print("\n" + "=" * 60)
print("[3단계] Random Forest 튜닝 (RandomizedSearchCV)")
print("=" * 60)

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 1.0],
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)

best_cv_rmse = np.sqrt(-search.best_score_)
print(f"\n튜닝 후 CV RMSE: {best_cv_rmse:.4f}")
print(f"최적 조합:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")


# ============================================================
# 4단계: 최종 평가
# ============================================================
print("\n" + "=" * 60)
print("[4단계] 최종 평가")
print("=" * 60)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  RMSE: {rmse:.4f}    (단위: $1000)")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"\n  베이스라인 대비 RMSE 개선: "
      f"{baseline_results['Random Forest']['test_rmse'] - rmse:+.3f}")


# ============================================================
# 5단계: 시각화
# ============================================================
print("\n" + "=" * 60)
print("[5단계] 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 베이스라인 비교
names = list(baseline_results.keys())
cv_rmses = [baseline_results[n]['cv_rmse'] for n in names]
axes[0, 0].barh(names, cv_rmses, color='lightblue')
axes[0, 0].set_xlabel('CV RMSE')
axes[0, 0].set_title('베이스라인 모델 비교')
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. 예측 vs 실제 (베스트 모델)
axes[0, 1].scatter(y_test, y_pred, alpha=0.5, color='steelblue')
axes[0, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('실제값')
axes[0, 1].set_ylabel('예측값')
axes[0, 1].set_title(f'예측 vs 실제 (튜닝된 Random Forest)\nR² = {r2:.4f}')
axes[0, 1].grid(alpha=0.3)

# 3. 잔차 분석
residuals = y_test - y_pred
axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('예측값')
axes[1, 0].set_ylabel('잔차')
axes[1, 0].set_title('잔차 플롯')
axes[1, 0].grid(alpha=0.3)

# 4. 특성 중요도
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns,
).sort_values()
importances.plot(kind='barh', ax=axes[1, 1], color='steelblue')
axes[1, 1].set_xlabel('중요도')
axes[1, 1].set_title('특성 중요도')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle('Boston Housing 회귀 — 종합 결과', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_summary.png', dpi=80)
plt.show()
print("저장: regression_summary.png")


# ============================================================
# 6단계: 모델 저장
# ============================================================
print("\n" + "=" * 60)
print("[6단계] 모델 저장")
print("=" * 60)

import joblib
import json

joblib.dump(best_model, 'boston_rf_tuned.pkl')
print("저장: boston_rf_tuned.pkl")

# 메타데이터
metadata = {
    'model_name': 'Random Forest Regressor',
    'task': 'regression (Boston Housing)',
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    },
    'hyperparameters': {k: int(v) if isinstance(v, np.integer) else
                           float(v) if isinstance(v, np.floating) else v
                       for k, v in search.best_params_.items()},
    'features': feature_names,
    'training_size': X_train.shape[0],
    'test_size': X_test.shape[0],
}

with open('boston_rf_tuned.metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("저장: boston_rf_tuned.metadata.json")

print("\n" + "=" * 60)
print("종합 실습 완료!")
print("=" * 60)
print("""
이 한 파일에서 다음을 다 했습니다:
1. 데이터 준비
2. 7가지 베이스라인 비교
3. 베스트 모델 자동 튜닝
4. 최종 평가 (RMSE/MAE/R²)
5. 4가지 시각화 (베이스라인, 예측, 잔차, 중요도)
6. 모델 + 메타데이터 저장

회사에서 회귀 문제를 받으면 이 패턴으로 시작하세요.
""")
