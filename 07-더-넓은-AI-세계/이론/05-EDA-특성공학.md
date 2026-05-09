# 7.5 EDA · 특성 공학 — 모델 전에 데이터를 알기

## 들어가며

좋은 데이터 + 평범한 모델 > 나쁜 데이터 + 최첨단 모델

이 부등식은 머신러닝 현장에서 거의 항상 성립합니다. 캐글 우승자들도 입을 모아 말합니다. "특성 공학에 80%, 모델 튜닝에 20%."

EDA(Exploratory Data Analysis, 탐색적 데이터 분석)는 모델을 만들기 전에 데이터를 이해하는 단계입니다. 특성 공학(Feature Engineering)은 그 이해를 바탕으로 모델이 학습하기 좋은 형태로 데이터를 가공하는 단계입니다.

이 장에서는 두 단계를 실전 흐름대로 짚어봅니다.

> 🧮 **수학 보충** — 분포, 평균, 분산, 상관계수, 분위수 같은 개념이 자주 등장합니다. [부록/수학/05-확률통계.md](../../부록/수학/05-확률통계.md)에서 한 번 훑어보시면 좋습니다.

---

## 1. EDA의 첫 단계 — 데이터 한눈에 파악하기

데이터를 처음 받으면 항상 같은 질문에서 시작합니다.

1. 행과 열은 몇 개인가?
2. 각 컬럼의 자료형은 무엇인가?
3. 결측값은 어디에 얼마나 있는가?
4. 수치 컬럼의 분포는 어떤가?
5. 범주 컬럼의 종류와 빈도는?
6. 타깃과 각 특성의 관계는?

이걸 5분 안에 끝내는 코드 패턴을 익혀두면 평생 씁니다.

### 1.1 첫 5줄 — 데이터 기초

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

print(df.shape)           # (행, 열)
df.head()                 # 처음 5행 미리보기
df.info()                 # 자료형과 결측 현황
df.describe()             # 수치 컬럼 통계 요약
df.describe(include="O")  # 범주 컬럼 통계
```

`info()`는 자료형이 의도한 대로인지(예: 숫자 컬럼이 object로 읽혔는지) 확인할 수 있어 중요합니다. `describe()`는 평균, 표준편차, 최소/최대, 사분위수를 한 번에 보여줍니다.

### 1.2 결측값 진단

```python
# 결측 개수와 비율
missing = pd.DataFrame({
    "count": df.isnull().sum(),
    "ratio": df.isnull().sum() / len(df) * 100
}).sort_values("ratio", ascending=False)
print(missing[missing["count"] > 0])
```

비율이 50%를 넘으면 컬럼 자체를 버리는 것을 고려합니다. 5% 이하라면 행을 버리거나 채우는 게 보통입니다.

### 1.3 중복 행

```python
print(df.duplicated().sum())
df = df.drop_duplicates()
```

대회 데이터는 깨끗하지만, 실무 데이터는 거의 항상 중복이 있습니다. 한 사용자가 여러 번 기록되거나, 시스템 오류로 같은 이벤트가 두 번 들어오는 경우입니다.

---

## 2. 분포 살피기 — 시각화의 힘

숫자만으로는 분포를 못 봅니다. matplotlib과 seaborn으로 한 줄씩 그려봅니다.

### 2.1 단변량 분포

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 수치 컬럼 — 히스토그램
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
num_cols = df.select_dtypes(include=np.number).columns[:6]
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()

# 범주 컬럼 — 막대그래프
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols[:3]:
    sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10])
    plt.xticks(rotation=45)
    plt.show()
```

이 단계에서 잡히는 문제들:
- 한쪽으로 심하게 치우친 분포 → 로그 변환 필요
- 봉우리가 두 개인 분포 → 숨은 그룹이 있을 수 있음
- 한 값에만 99%가 몰린 컬럼 → 거의 정보가 없으니 제거 고려
- 범주가 100개 이상인 컬럼 → 빈도 낮은 것을 "기타"로 묶기

### 2.2 이상치(outlier) 찾기

```python
# 박스플롯
fig, axes = plt.subplots(1, len(num_cols), figsize=(4*len(num_cols), 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()

# IQR 기준 이상치 개수
def count_outliers(s):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return ((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum()

for col in num_cols:
    print(f"{col:20s}  outliers: {count_outliers(df[col])}")
```

이상치는 진짜 오류일 수도, 의미 있는 신호일 수도 있습니다. 사기 거래 탐지에서는 이상치 자체가 타깃이죠. 무조건 제거하지 말고 출처를 확인하세요.

### 2.3 타깃과의 관계

분류 문제라면 클래스별로 특성 분포를 비교합니다.

```python
target = "is_churn"   # 가정: 이탈 여부 (0/1)

for col in num_cols:
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"{col} by {target}")
    plt.show()
```

회귀 문제라면 산점도를 그립니다.

```python
target = "price"
for col in num_cols:
    sns.scatterplot(data=df, x=col, y=target, alpha=0.3)
    plt.title(f"{col} vs {target}")
    plt.show()
```

여기서 강한 신호를 보이는 특성은 모델에 큰 도움이 됩니다.

### 2.4 상관 행렬

```python
corr = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.show()
```

피어슨 상관은 선형 관계만 잡습니다. 비선형 관계가 의심되면 스피어만 상관(`df.corr(method="spearman")`)도 함께 보세요.

상관이 0.9 이상인 특성 쌍은 **다중공선성** 문제를 일으킬 수 있어, 둘 중 하나를 제거하거나 PCA로 합치는 것을 고려합니다.

---

## 3. 결측값 처리 — 상황별 전략

결측값은 단순히 채우는 게 아닙니다. **왜 비어 있는가**에 따라 전략이 달라집니다.

### 3.1 결측의 세 종류

| 종류 | 의미 | 예시 |
|------|------|------|
| MCAR (완전 무작위) | 결측이 어떤 변수와도 무관 | 센서가 우연히 한 번 빠짐 |
| MAR (조건부 무작위) | 다른 변수에 따라 결측 발생 | 남성이 체중을 덜 입력 |
| MNAR (체계적) | 값 자체가 결측 원인 | 고소득자가 소득을 안 적음 |

MNAR이 가장 까다롭습니다. 결측 자체가 정보이므로, "결측이었음" 표시 컬럼을 새로 만드는 게 좋습니다.

### 3.2 처리 방법

```python
from sklearn.impute import SimpleImputer, KNNImputer

# 방법 1: 평균/중앙값 대체 (수치)
imputer = SimpleImputer(strategy="median")  # 이상치에 강함
df[num_cols] = imputer.fit_transform(df[num_cols])

# 방법 2: 최빈값 대체 (범주)
imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imputer.fit_transform(df[cat_cols])

# 방법 3: KNN 대체 (다른 행 참고)
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

# 방법 4: 결측 표시 컬럼 추가 + 단순 대체
df["age_was_missing"] = df["age"].isnull().astype(int)
df["age"] = df["age"].fillna(df["age"].median())
```

**팁**: 평균은 이상치에 끌려가니, 중앙값이 보통 더 안전합니다. 그리고 train의 통계로 채우고, test에는 같은 값을 적용해야 정보 누수가 없습니다.

```python
# 올바른 패턴
median = train_df["age"].median()   # train에서 계산
train_df["age"] = train_df["age"].fillna(median)
test_df["age"] = test_df["age"].fillna(median)   # 같은 값을 test에 적용
```

---

## 4. 범주형 변수 인코딩

ML 모델은 숫자만 받으니 범주를 숫자로 바꿔야 합니다. 어떻게 바꾸느냐가 성능을 좌우합니다.

### 4.1 One-Hot Encoding

```python
df = pd.get_dummies(df, columns=["color", "size"], drop_first=True)
# color_red, color_blue, size_M, size_L 등의 컬럼 생성
```

장점: 모든 모델에서 잘 작동
단점: 카테고리 수만큼 컬럼이 늘어남. 카디널리티(고유값 수)가 100을 넘으면 비효율적

### 4.2 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])
# A=0, B=1, C=2, ...
```

장점: 컬럼 수 안 늘어남
단점: 트리 기반 모델(Random Forest, XGBoost)에는 OK, **선형 모델/신경망에는 위험**. "A=0, B=1, C=2"는 "C가 A의 두 배"라는 가짜 순서를 만듭니다.

### 4.3 Target Encoding

각 카테고리를 그 카테고리의 평균 타깃값으로 대체합니다.

```python
# 직접 구현
target_mean = df.groupby("city")["price"].mean()
df["city_encoded"] = df["city"].map(target_mean)
```

장점: 카디널리티 높아도 컬럼 1개로 처리
단점: 정보 누수 위험 큼. **K-Fold target encoding**으로 처리해야 함

```python
from sklearn.model_selection import KFold

def kfold_target_encode(train, test, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    train[f"{col}_te"] = np.nan
    for tr_idx, val_idx in kf.split(train):
        means = train.iloc[tr_idx].groupby(col)[target].mean()
        train.iloc[val_idx, train.columns.get_loc(f"{col}_te")] = (
            train.iloc[val_idx][col].map(means)
        )
    test[f"{col}_te"] = test[col].map(train.groupby(col)[target].mean())
    return train, test
```

복잡하지만, 카디널리티가 높은 범주(우편번호, 사용자 ID, 상품 ID 등)에는 큰 효과가 있습니다.

### 4.4 Ordinal Encoding (순서 있는 범주)

```python
order = ["low", "medium", "high"]
mapping = {v: i for i, v in enumerate(order)}
df["level_encoded"] = df["level"].map(mapping)
```

학력, 등급처럼 자연스러운 순서가 있는 경우에 적절합니다.

---

## 5. 수치형 변수 변환 — 분포를 모델 친화적으로

### 5.1 스케일링

선형 모델, KNN, 신경망은 특성 간 스케일이 다르면 학습이 불안정합니다. 트리 기반은 영향을 안 받습니다.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 표준화: 평균 0, 표준편차 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 정규화: 0~1 범위
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 강건한 스케일링: 중앙값과 IQR 사용 (이상치 강함)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

신경망에는 표준화가 가장 흔합니다. 이미지처럼 0~255 범위가 명확하면 MinMax도 좋습니다.

### 5.2 로그 변환

수입, 가격, 카운트처럼 우측 꼬리가 긴 분포에 효과적입니다.

```python
df["log_income"] = np.log1p(df["income"])  # log(1+x), 0 처리 가능
```

타깃도 변환할 수 있습니다. 회귀에서 가격을 예측한다면 `np.log1p(price)`로 학습하고, 예측 후 `np.expm1`으로 되돌립니다.

### 5.3 비닝 (Binning)

연속값을 구간으로 묶습니다. 비선형 관계를 단순한 모델로 잡고 싶을 때 유용합니다.

```python
# 등간격 구간
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                       labels=["청소년", "청년", "중년", "노년"])

# 등빈도 구간
df["income_decile"] = pd.qcut(df["income"], q=10, labels=False)
```

### 5.4 이상치 처리

완전 제거보다 **클리핑(winsorizing)**이 보통 더 안전합니다.

```python
lower, upper = df["price"].quantile([0.01, 0.99])
df["price_clipped"] = df["price"].clip(lower, upper)
```

상위/하위 1%를 그 경계값으로 누르는 방식입니다.

---

## 6. 새 특성 만들기 — 도메인 지식의 가치

가장 임팩트가 큰 단계입니다. 여기서 캐글 순위가 갈립니다.

### 6.1 날짜/시간 분해

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek    # 0=월요일
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["quarter"] = df["date"].dt.quarter
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
```

### 6.2 주기성 인코딩

월(1~12), 시간(0~23), 요일(0~6)은 순환합니다. 1월과 12월은 숫자상 멀지만 의미상 가깝죠. 사인/코사인으로 인코딩합니다.

```python
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

### 6.3 그룹별 통계 (Aggregation features)

각 사용자의 평균, 표준편차, 최댓값 등을 행 단위로 붙입니다.

```python
user_stats = df.groupby("user_id").agg({
    "amount": ["mean", "std", "max", "count"],
    "category": "nunique"
}).reset_index()
user_stats.columns = ["user_id", "amt_mean", "amt_std",
                      "amt_max", "amt_count", "cat_diversity"]
df = df.merge(user_stats, on="user_id", how="left")
```

이런 특성은 사용자/상품/지역 단위 패턴을 모델에 제공합니다.

### 6.4 비율과 차이

원시 컬럼 두 개의 비율이 더 강력한 신호인 경우가 많습니다.

```python
df["price_per_sqm"] = df["price"] / df["area"]
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df["debt_to_income"] = df["debt"] / df["income"]
```

### 6.5 교차 특성

두 컬럼을 결합해 새 특성을 만듭니다.

```python
df["city_x_age"] = df["city"] + "_" + df["age_bin"].astype(str)
# "Seoul_청년", "Busan_노년" 같은 새 카테고리
```

신경망은 이런 교차를 자동으로 학습하지만, 트리 모델은 직접 만들어주면 도움이 됩니다.

### 6.6 텍스트에서 추출

```python
df["title_length"] = df["title"].str.len()
df["title_word_count"] = df["title"].str.split().str.len()
df["has_question"] = df["title"].str.contains(r"\?").astype(int)
df["uppercase_ratio"] = df["title"].apply(
    lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
)
```

---

## 7. 특성 선택 — 노이즈 빼기

특성이 많다고 항상 좋은 건 아닙니다. 무관한 특성은 노이즈를 더하고 과적합을 유발합니다.

### 7.1 분산 기반 — 거의 안 변하는 특성 제거

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

### 7.2 단변량 통계 — 타깃과의 관계 강한 것 선택

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

상호정보량(mutual information)은 비선형 관계도 잡습니다.

### 7.3 모델 기반 — 트리의 feature_importance

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(20).plot(kind="barh")
```

상위 N개만 선택하거나, 임계치 이하를 제거합니다.

### 7.4 재귀 특성 제거 (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

가장 영향력 적은 특성을 한 개씩 제거하며 반복합니다. 시간은 걸리지만 효과적입니다.

---

## 8. 정보 누수 — 가장 흔한 실수

특성 공학에서 발생하는 정보 누수 사례들입니다.

### 8.1 미래 정보 사용

```python
# 잘못된 예: 전체 데이터의 평균을 사용
df["price_norm"] = df["price"] / df["price"].mean()
# train과 test가 섞여서 test 정보가 train으로 새어들어감
```

### 8.2 타깃을 직접 쓴 특성

```python
# 잘못된 예
df["is_high_value"] = (df["target"] > df["target"].median()).astype(int)
# 이건 그냥 타깃을 변형한 거라 사용 불가
```

### 8.3 시계열에서 미래 윈도우

```python
# 잘못된 예: 과거+미래 평균
df["rolling_mean"] = df["sales"].rolling(7, center=True).mean()

# 올바른 예: 과거만 사용
df["rolling_mean"] = df["sales"].rolling(7).mean().shift(1)
```

### 8.4 올바른 파이프라인 — sklearn Pipeline

이런 실수를 막는 가장 안전한 방법은 sklearn의 `Pipeline`을 쓰는 것입니다.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_features = ["age", "income", "amount"]
cat_features = ["city", "category"]

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

full_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100)),
])

full_pipeline.fit(X_train, y_train)
print(full_pipeline.score(X_test, y_test))
```

이 패턴이면 train의 통계를 자동으로 test에 적용합니다. cross-validation에서도 fold마다 올바르게 처리됩니다.

---

## 9. 실전 흐름 — 처음부터 끝까지

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. 로드와 빠른 점검 ---
df = pd.read_csv("data.csv")
print(df.shape, df.dtypes.value_counts())
print("결측 비율:", (df.isnull().sum() / len(df)).sort_values(ascending=False).head())

# --- 2. 타깃 분리 ---
y = df["target"]
X = df.drop(columns=["target"])

# --- 3. 도메인 기반 새 특성 ---
X["date"] = pd.to_datetime(X["date"])
X["dayofweek"] = X["date"].dt.dayofweek
X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
X = X.drop(columns=["date"])

# --- 4. 컬럼 분류 ---
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

# --- 5. 파이프라인 ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ]), cat_cols),
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingClassifier(random_state=0)),
])

# --- 6. 교차 검증으로 평가 ---
scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# --- 7. 최종 학습 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
pipeline.fit(X_train, y_train)
print("Test AUC:", pipeline.score(X_test, y_test))
```

이 템플릿이 베이스라인입니다. 여기에 도메인 특성을 더하고, 모델을 바꿔가며 점진적으로 개선하는 게 일반적인 흐름입니다.

---

## 10. 한눈에 정리

**EDA 체크리스트**
- [ ] 행/열 수, 자료형 확인
- [ ] 결측값 진단 (양과 패턴)
- [ ] 중복 행 처리
- [ ] 단변량 분포 (히스토그램, 박스플롯)
- [ ] 타깃과의 관계 (KDE, 산점도)
- [ ] 상관 행렬

**특성 공학 체크리스트**
- [ ] 결측 대체 (train 기준)
- [ ] 범주 인코딩 (모델에 맞게)
- [ ] 수치 스케일링 (필요시)
- [ ] 로그 변환 (치우친 분포)
- [ ] 도메인 특성 추가 (날짜 분해, 비율, 그룹 통계)
- [ ] 정보 누수 체크
- [ ] sklearn Pipeline으로 묶기

---

## 더 깊이 가고 싶으시면

- **AutoML 도구**: AutoGluon, H2O AutoML, FLAML — 특성 공학과 모델 선택 자동화
- **Featuretools**: 관계형 데이터에서 자동 특성 생성
- **shap, eli5**: 특성 중요도 해석 (다음 장에서 다룹니다)
- **`pandas-profiling`, `sweetviz`, `ydata-profiling`**: EDA 자동 보고서
- **카테고리 임베딩**: 신경망에서 학습 가능한 범주 인코딩
- **Time Series 특성**: lag, rolling window, Fourier features 등 시계열 전용

다음 장에서는 학습된 모델이 "왜 그렇게 예측했는지" 들여다보는 **모델 해석**을 다룹니다. SHAP, LIME, feature importance 같은 도구들입니다.
