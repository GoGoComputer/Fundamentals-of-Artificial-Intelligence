"""2장 보너스 실습: 타이타닉 생존 예측

MNIST는 픽셀(이미지)이지만, 실무에서 더 자주 만나는 건
'표 형태의 데이터'예요. 타이타닉으로 한 번 해 봅시다.

"이 사람이 타이타닉 침몰에서 살아남았을까 못 살아남았을까?"
이진 분류 문제입니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import randint, uniform


# ============================================================
# 1. 데이터 불러오기
# ============================================================
# seaborn에 내장된 타이타닉 데이터
df = sns.load_dataset('titanic')

print(f"데이터 모양: {df.shape}")
print(f"\n첫 5행:")
print(df.head())

print(f"\n각 열 정보:")
print(df.info())


# ============================================================
# 2. 데이터 탐색
# ============================================================
print(f"\n생존자 비율: {df['survived'].mean():.2%}")
print(f"\n결측값:")
print(df.isnull().sum().sort_values(ascending=False).head(10))


# ============================================================
# 3. 전처리
# ============================================================
# 사용할 특성 선택
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features].copy()
y = df['survived'].copy()

# 결측값 처리
X['age'] = X['age'].fillna(X['age'].median())
X['embarked'] = X['embarked'].fillna(X['embarked'].mode()[0])
X['fare'] = X['fare'].fillna(X['fare'].median())

# 범주형 → 숫자 (one-hot encoding)
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

print(f"\n전처리 후 특성: {X.columns.tolist()}")
print(f"X 모양: {X.shape}")


# ============================================================
# 4. 분할
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습: {X_train.shape}, 평가: {X_test.shape}")


# ============================================================
# 5. 두 모델 비교
# ============================================================

# Pipeline으로 정규화 + 모델 묶기
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]),
}

results = {}
for name, model in models.items():
    print(f"\n[{name}]")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  정확도: {acc:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")

    results[name] = {'acc': acc, 'auc': auc, 'model': model, 'y_pred': y_pred}


# ============================================================
# 6. 베스트 모델 자세히
# ============================================================
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n[베스트: {best_name}]")
print(classification_report(y_test, best['y_pred']))


# ============================================================
# 7. 특성 중요도 (Random Forest인 경우)
# ============================================================
if best_name == "Random Forest":
    rf = best['model'].named_steps['clf']
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='steelblue')
    plt.xlabel('중요도')
    plt.title('Random Forest 특성 중요도')
    plt.tight_layout()
    plt.savefig('titanic_importance.png', dpi=80)
    plt.show()
    print("저장: titanic_importance.png")
    print("\n→ 어떤 특성이 생존에 영향이 컸는지 보세요")


# ============================================================
# 8. 혼동 행렬
# ============================================================
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['사망 예측', '생존 예측'],
    yticklabels=['실제 사망', '실제 생존'],
)
plt.title(f'{best_name} 혼동 행렬')
plt.tight_layout()
plt.savefig('titanic_cm.png', dpi=80)
plt.show()


# ============================================================
# 9. 새 사람 한 명 예측 (재미)
# ============================================================
print("\n[9] 새 사람 예측해 보기")
print("-" * 40)

# 한 사람의 정보 (1등석 여성, 30세 등)
new_person = pd.DataFrame({
    'pclass': [1],
    'age': [30],
    'sibsp': [0],
    'parch': [0],
    'fare': [80.0],
    'sex_male': [False],     # 여성
    'embarked_Q': [False],
    'embarked_S': [True],    # 사우샘프턴
})

# 학습 데이터의 컬럼 순서 맞추기
new_person = new_person.reindex(columns=X.columns, fill_value=0)

prob = best['model'].predict_proba(new_person)[0, 1]
print(f"30세 여성, 1등석, 단독 탑승")
print(f"생존 확률: {prob:.2%}")

if prob > 0.5:
    print("→ 살아남았을 가능성이 큽니다 (당시 여성+상류층 우선 구조)")
else:
    print("→ 살아남기 어려웠을 가능성이 큽니다")


print("\n타이타닉 분류 끝! MNIST보다 표 데이터가 익숙해지셨을 거예요.")
