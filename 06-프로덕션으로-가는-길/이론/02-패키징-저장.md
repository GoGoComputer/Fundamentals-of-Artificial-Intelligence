# 6.2 모델 패키징과 저장

## "그냥 model.save()로 충분 아닌가요?"

이게 가장 자주 만나는 함정이에요. 모델 가중치만 저장하면 **운영에서 잘못된 예측이 나옵니다.** 왜냐하면:

```python
# 학습 시
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y)
torch.save(model.state_dict(), 'model.pth')

# 운영 시
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
prediction = model(new_data)    # ❌ 정규화 안 했음!
```

`scaler`가 어디 갔어요? 운영 단계에서는 학습 시 썼던 정규화 정보가 없어요. 그래서 잘못된 예측이 나옵니다.

이 글에서는 **모델 + 전처리 + 메타데이터를 한 묶음으로 저장**하는 법을 배웁니다.

---

## 무엇을 저장해야 하나요?

체크리스트:

```
[ ] 모델 가중치
[ ] 모델 아키텍처 (어떤 구조인지)
[ ] 전처리 객체 (Scaler, Encoder 등)
[ ] 특성 이름과 순서
[ ] 학습에 쓰인 라이브러리 버전
[ ] 학습 시 메타데이터 (날짜, 데이터 해시, 성능 등)
[ ] 사용 예시 코드
```

이걸 다 묶어서 한 폴더로 관리하는 게 좋아요.

---

## 패키징 패턴

### 디렉토리 구조

```
my_model_v1.0.0/
├── model.pth                    # 모델 가중치
├── model_architecture.py        # 모델 클래스 정의
├── preprocessor.pkl             # Scaler, Encoder 등
├── metadata.json                # 메타데이터
├── requirements.txt             # 라이브러리 버전
├── README.md                    # 사용법
└── example_usage.py             # 사용 예시
```

### 한꺼번에 저장하기

가장 단순한 방법: **모든 걸 dict에 담아 pickle로 저장.**

```python
import pickle
import json
from datetime import datetime
import sklearn

bundle = {
    'model_state_dict': model.state_dict(),
    'preprocessor': scaler,
    'feature_names': feature_names,
    'metadata': {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'training_data_size': len(X_train),
        'test_metrics': {
            'rmse': 3.12,
            'r2': 0.84,
        },
        'sklearn_version': sklearn.__version__,
    },
}

with open('my_model.pkl', 'wb') as f:
    pickle.dump(bundle, f)
```

불러올 때:

```python
with open('my_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = MyModel()
model.load_state_dict(bundle['model_state_dict'])

scaler = bundle['preprocessor']

# 추론 시
new_data_scaled = scaler.transform(new_data)
prediction = model(torch.tensor(new_data_scaled, dtype=torch.float32))
```

---

## sklearn은 더 쉬워요: Pipeline

sklearn의 `Pipeline`을 쓰면 전처리 + 모델을 자동으로 한 묶음으로 관리할 수 있어요.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# 학습 시
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.pkl')


# 운영 시
pipeline = joblib.load('pipeline.pkl')
prediction = pipeline.predict(new_data)    # 자동으로 scaler 적용됨!
```

이게 sklearn 모델 운영의 표준 패턴이에요. **항상 Pipeline 쓰세요.**

---

## PyTorch 모델 저장 형식

PyTorch에서 모델을 저장하는 방법은 여러 가지가 있어요.

### 1. `state_dict` 저장 (권장)

```python
# 저장
torch.save(model.state_dict(), 'model.pth')

# 불러오기 (먼저 같은 클래스로 인스턴스 만들기)
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

- ✅ 가장 안전 (PyTorch 버전 독립)
- ✅ 권장 방식
- ❌ 모델 클래스 정의가 코드에 있어야 함

### 2. 모델 통째로 저장 (비권장)

```python
torch.save(model, 'full_model.pth')

model = torch.load('full_model.pth')
```

- ✅ 클래스 정의 없이 불러올 수 있음
- ❌ PyTorch 버전이 바뀌면 에러
- ❌ 클래스 위치가 바뀌면 에러
- ❌ 보안 위험

### 3. TorchScript (배포용)

학습은 PyTorch로, 배포는 PyTorch 없이. C++/모바일에서도 동작.

```python
model.eval()
example = torch.rand(1, 13)
traced = torch.jit.trace(model, example)
traced.save('model_traced.pt')

# 불러오기 (PyTorch 없이도 됨)
loaded = torch.jit.load('model_traced.pt')
output = loaded(input_tensor)
```

### 4. ONNX (다른 프레임워크 호환)

```python
model.eval()
example = torch.rand(1, 13)

torch.onnx.export(
    model, example, 'model.onnx',
    input_names=['input'],
    output_names=['output'],
)
```

ONNX 형식은 TensorFlow, MXNet, 모바일, 브라우저 등 다양한 곳에서 동작.

### 비교 표

| 형식 | 사용 |
|------|------|
| `.pth` (state_dict) | PyTorch 환경에서 사용 (학습/연구) |
| `.pt` (TorchScript) | C++ / 모바일 배포 |
| `.onnx` | 다른 프레임워크 / 브라우저 |
| `.pt2` (torch.compile) | 최적화된 추론 |

---

## 메타데이터의 중요성

모델 파일만 있으면 한 달 뒤에 본인이 봐도 모르는 게 됩니다. **메타데이터를 같이 저장하세요.**

```python
import json
import hashlib
from datetime import datetime

metadata = {
    # 기본 정보
    'name': 'boston_house_price_predictor',
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'created_by': 'ml-team',
    
    # 모델 정보
    'model_type': 'RandomForestRegressor',
    'framework': 'sklearn',
    'framework_version': sklearn.__version__,
    
    # 데이터 정보
    'training_data': {
        'size': len(X_train),
        'features': feature_names,
        'target': 'house_price',
        'date_range': '2020-01-01 to 2024-12-31',
        'data_hash': hashlib.md5(X_train.tobytes()).hexdigest(),
    },
    
    # 성능
    'metrics': {
        'rmse': 3.12,
        'mae': 2.45,
        'r2': 0.84,
    },
    
    # 하이퍼파라미터
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 20,
        'random_state': 42,
    },
    
    # 입력 스펙 (운영용)
    'input_spec': {
        'feature_1': {'type': 'float', 'range': [0, 100]},
        'feature_2': {'type': 'int', 'range': [0, 50]},
        # ...
    },
    
    # 출력 스펙
    'output_spec': {
        'type': 'float',
        'unit': 'USD * 1000',
        'range': [0, 1000],
    },
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
```

이런 메타데이터가 있으면:
- 이 모델이 언제, 어떤 데이터로 학습됐는지 알 수 있음
- API 문서를 자동 생성할 수 있음
- 입력 검증을 자동화할 수 있음

---

## 모델 버전 관리

### 단순한 방법: 폴더 + 시맨틱 버저닝

```
models/
├── boston_v1.0.0/
│   ├── model.pkl
│   └── metadata.json
├── boston_v1.0.1/
│   └── ...
└── boston_v1.1.0/    # 메이저 변경 (신규 특성 추가 등)
    └── ...
```

### 시맨틱 버저닝

```
v1.0.0
 │ │ │
 │ │ └─ 패치: 작은 버그 수정
 │ └─── 마이너: 호환되는 기능 추가
 └───── 메이저: 호환 깨지는 변경
```

ML에서는 보통:
- **메이저**: 입력 특성이 바뀜
- **마이너**: 모델 알고리즘 바뀜
- **패치**: 데이터만 새로 학습

### 도구 사용

위처럼 직접 관리해도 되지만, 도구가 더 편해요.

| 도구 | 특징 |
|------|------|
| **MLflow Models** | sklearn/PyTorch 등 다양한 형식 지원 |
| **DVC** | git처럼 모델 + 데이터 버전 관리 |
| **W&B Artifacts** | 실험 추적과 통합 |

---

## 실전 예: 풀 패키지 만들기

```python
import os
import json
import pickle
import torch
from datetime import datetime


def save_model_bundle(
    model,
    preprocessor,
    feature_names,
    metrics,
    output_dir,
    version='1.0.0',
):
    """모델을 번들로 저장."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 모델 가중치
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    else:
        with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    # 2. 전처리기
    with open(os.path.join(output_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # 3. 메타데이터
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'feature_names': feature_names,
        'metrics': metrics,
        'model_type': type(model).__name__,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 4. requirements.txt
    requirements = """
torch==2.1.0
scikit-learn==1.3.0
numpy==1.26.0
pandas==2.1.0
""".strip()
    with open(os.path.join(output_dir, 'requirements.txt'), 'w') as f:
        f.write(requirements)
    
    # 5. README.md
    readme = f"""# {metadata.get('name', 'Model')} v{version}

## 사용법

```python
import pickle
import torch

# 전처리기 불러오기
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# 모델 불러오기 (구조는 따로 정의 필요)
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 예측
features = ...    # 입력
features_scaled = preprocessor.transform(features)
with torch.no_grad():
    prediction = model(torch.tensor(features_scaled, dtype=torch.float32))
```

## 성능

{json.dumps(metrics, indent=2)}

## 특성

{', '.join(feature_names)}
"""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme)
    
    print(f"모델 번들 저장 완료: {output_dir}/")


# 사용
save_model_bundle(
    model=trained_model,
    preprocessor=scaler,
    feature_names=['feature_1', 'feature_2', ...],
    metrics={'rmse': 3.12, 'r2': 0.84},
    output_dir='models/boston_v1.0.0',
    version='1.0.0',
)
```

---

## 모델 불러오기 (운영용)

```python
import pickle
import json
import torch
from pathlib import Path


def load_model_bundle(bundle_dir):
    """저장된 모델 번들 불러오기."""
    bundle_dir = Path(bundle_dir)
    
    # 메타데이터
    with open(bundle_dir / 'metadata.json') as f:
        metadata = json.load(f)
    
    # 전처리기
    with open(bundle_dir / 'preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # 모델 (PyTorch라면 클래스가 미리 import 돼 있어야 함)
    if (bundle_dir / 'model.pth').exists():
        from my_model_classes import MyModel    # 미리 정의돼야
        model = MyModel()
        model.load_state_dict(torch.load(bundle_dir / 'model.pth'))
        model.eval()
    else:
        with open(bundle_dir / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    return model, preprocessor, metadata


# 사용
model, scaler, metadata = load_model_bundle('models/boston_v1.0.0')
print(f"모델 버전: {metadata['version']}")
print(f"성능: {metadata['metrics']}")
```

---

## 정리

```
좋은 모델 패키지:
├── model.pth (또는 .pkl)         # 가중치
├── preprocessor.pkl              # Scaler 등
├── metadata.json                 # 모든 정보
├── requirements.txt              # 의존성
└── README.md                     # 사용법

핵심 원칙:
- Pipeline (sklearn) 또는 dict bundle (PyTorch)로 묶기
- 메타데이터 항상 같이
- 버전 관리 (v1.0.0 형식)
- 의존성 명시
```

➡️ 다음: [6.3 FastAPI로 모델 서빙](03-FastAPI-서빙.md)
