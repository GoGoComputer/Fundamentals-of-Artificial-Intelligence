"""6장 1~2절 실습: 모델 패키징과 저장

학습된 모델을 운영용 패키지로 만듭니다.
이 패키지는 다음 실습(FastAPI)에서 사용해요.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. 데이터 + 학습
# ============================================================
print("[1] 데이터 준비 + 모델 학습")

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

X = pd.DataFrame(data, columns=feature_names)
y = pd.Series(target, name='MEDV')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)


# ============================================================
# 2. Pipeline (정규화 + 모델 묶기)
# ============================================================
print("\n[2] Pipeline 구성")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)


# ============================================================
# 3. 평가
# ============================================================
y_pred = pipeline.predict(X_test)
metrics = {
    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    'mae': float(mean_absolute_error(y_test, y_pred)),
    'r2': float(r2_score(y_test, y_pred)),
}
print(f"\n성능: {metrics}")


# ============================================================
# 4. 저장 (운영용 번들)
# ============================================================
print("\n[3] 모델 번들 저장")

VERSION = '1.0.0'
output_dir = f'models/boston_v{VERSION}'
os.makedirs(output_dir, exist_ok=True)


# 4-1. Pipeline 저장
joblib.dump(pipeline, os.path.join(output_dir, 'pipeline.pkl'))


# 4-2. 메타데이터
metadata = {
    'name': 'boston_house_price_predictor',
    'version': VERSION,
    'created_at': datetime.now().isoformat(),
    'task': 'regression',

    'model_info': {
        'type': 'RandomForestRegressor',
        'framework': 'scikit-learn',
        'preprocessor': 'StandardScaler',
        'n_estimators': 200,
        'max_depth': 20,
    },

    'data_info': {
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': feature_names,
        'target': 'MEDV (house price * $1000)',
    },

    'metrics': metrics,

    'input_spec': {
        feature: {'type': 'float', 'description': '...'}
        for feature in feature_names
    },

    'output_spec': {
        'type': 'float',
        'unit': 'USD * 1000',
        'min': 0,
    },
}

with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)


# 4-3. requirements.txt
with open(os.path.join(output_dir, 'requirements.txt'), 'w') as f:
    f.write("""scikit-learn==1.3.0
numpy==1.26.0
pandas==2.1.0
joblib==1.3.0
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.0
""")


# 4-4. README.md
readme_content = f"""# Boston House Price Predictor v{VERSION}

## 설명
보스턴 지역 집값 예측 모델.

## 성능
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- R²: {metrics['r2']:.4f}

## 사용법

```python
import joblib

pipeline = joblib.load('pipeline.pkl')

import pandas as pd
features = pd.DataFrame({{
    'CRIM': [0.00632],
    'ZN': [18.0],
    # ... 13개 특성
}})

prediction = pipeline.predict(features)
print(prediction)
```

## 입력 특성
{', '.join(feature_names)}

## 학습 정보
- 학습일: {metadata['created_at']}
- 학습 데이터 크기: {metadata['data_info']['training_samples']}
"""

with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_content)


# 4-5. example_usage.py
example = '''"""사용 예시 — 이대로 실행하면 됨."""
import joblib
import pandas as pd

# 모델 불러오기
pipeline = joblib.load('pipeline.pkl')

# 예측할 집의 정보
example_house = pd.DataFrame({
    'CRIM': [0.00632],
    'ZN': [18.0],
    'INDUS': [2.31],
    'CHAS': [0],
    'NOX': [0.538],
    'RM': [6.575],
    'AGE': [65.2],
    'DIS': [4.0900],
    'RAD': [1.0],
    'TAX': [296.0],
    'PTRATIO': [15.3],
    'B': [396.90],
    'LSTAT': [4.98],
})

prediction = pipeline.predict(example_house)
print(f"예측 집값: ${prediction[0]*1000:.0f}")
'''

with open(os.path.join(output_dir, 'example_usage.py'), 'w') as f:
    f.write(example)


print(f"\n저장 완료! 폴더: {output_dir}/")
print("파일들:")
for fname in sorted(os.listdir(output_dir)):
    print(f"  - {fname}")


# ============================================================
# 5. 불러와서 사용해 보기 (검증)
# ============================================================
print(f"\n[4] 저장된 모델 불러와서 검증")

loaded_pipeline = joblib.load(os.path.join(output_dir, 'pipeline.pkl'))

with open(os.path.join(output_dir, 'metadata.json'), encoding='utf-8') as f:
    loaded_metadata = json.load(f)

# 같은 X_test로 같은 결과가 나와야 함
y_pred_loaded = loaded_pipeline.predict(X_test)

is_same = np.allclose(y_pred, y_pred_loaded)
print(f"원본과 불러온 모델의 예측이 같은가? {is_same}")

print(f"\n메타데이터: 버전 {loaded_metadata['version']}, "
      f"R² {loaded_metadata['metrics']['r2']:.4f}")


print("\n[5] 다음 실습에서 이 모델로 API를 만들어 봅시다!")
