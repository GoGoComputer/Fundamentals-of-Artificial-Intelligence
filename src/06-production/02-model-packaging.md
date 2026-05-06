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

➡️ 다음: [6.3 FastAPI로 모델 서빙](03-fastapi-serving.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **6.2 모델 패키징과 저장** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **6.2 모델 패키징과 저장**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

---

## 자주 하는 오해와 바로잡기

### 1) "이론만 이해하면 실습은 저절로 된다"
아닙니다. 이론 이해와 코드 실행 능력은 별개의 근육입니다. 둘을 같이 써야 합니다.

### 2) "예제 코드가 돌아갔으니 완전히 이해했다"
동작 확인은 시작일 뿐입니다. 파라미터를 바꿨을 때 결과가 어떻게 달라지는지까지 확인해야 이해가 완성됩니다.

### 3) "좋은 모델은 항상 하나다"
데이터 특성과 제약(시간, 메모리, 응답 속도)에 따라 최선의 선택은 달라집니다.

### 4) "지표 숫자만 높으면 끝이다"
지표는 해석이 필요합니다. 어떤 데이터에서, 어떤 분포에서, 어떤 기준으로 얻은 점수인지 함께 보아야 합니다.

### 5) "지금은 초반이니까 배포는 나중에 생각해도 된다"
초반부터 재현성과 저장 구조를 습관화해야 나중에 배포 단계에서 무너지지 않습니다.

---

## 실전 연결 시나리오

아래는 이 단원 내용을 실제 업무로 연결할 때 자주 쓰는 흐름입니다.

1. 문제를 한 문장으로 정의한다.  
예: "6.2 모델 패키징과 저장 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

2. 성공 기준을 지표로 정한다.  
예: 정확도/RMSE/F1/지연시간 같은 운영 가능한 숫자로 정의.

3. 가장 단순한 베이스라인을 만든다.  
복잡한 모델보다 먼저, 빠르게 재현 가능한 기본 모델을 세웁니다.

4. 한 번에 하나씩 개선한다.  
전처리, 모델, 튜닝, 임계값, 배치 크기 등은 동시에 바꾸지 않고 순차적으로 바꿉니다.

5. 결과를 로그와 함께 저장한다.  
실험 이름, 파라미터, 점수, 데이터 버전을 함께 남겨야 재현이 됩니다.

6. 실패 케이스를 분석한다.  
점수 평균보다 오분류/고오차 샘플을 분석할 때 개선 아이디어가 나옵니다.

7. 배포 가능 형태로 정리한다.  
모델 파일, 메타데이터, 입력 스키마, 테스트 스크립트까지 묶습니다.

---

## 복습 체크리스트

다음 질문에 답할 수 있으면 이 단원을 실전 수준으로 이해한 것입니다.

- 이 단원의 핵심 개념을 중학생에게 설명하듯 말할 수 있는가?
- 코드의 각 블록이 왜 필요한지 한 줄씩 설명할 수 있는가?
- 이 단원에서 가장 자주 틀리는 지점을 알고 있는가?
- 지표를 볼 때 어떤 함정이 있는지 알고 있는가?
- 베이스라인과 개선 모델의 차이를 표로 정리할 수 있는가?
- 재현 가능하게 실험을 저장하는 습관이 있는가?
- 실패 사례를 보고 다음 실험 가설을 만들 수 있는가?
- 이 단원 내용을 다음 장과 어떻게 연결할지 설명할 수 있는가?

---

## 확장 연습 과제

### 과제 A: 파라미터 감각 만들기
현재 예제에서 핵심 파라미터 1~2개만 바꿔 5회 실험하고, 결과를 표로 정리해 보세요.

### 과제 B: 실패 사례 분석
오분류 또는 오차가 큰 샘플 20개를 모아 공통 패턴을 찾아 보세요.

### 과제 C: 설명 가능 리포트 작성
"무엇을 했고, 왜 했고, 결과가 어땠고, 다음엔 무엇을 할지"를 1페이지로 정리해 보세요.

이 3개를 직접 해 보면 **6.2 모델 패키징과 저장**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

6.2 모델 패키징과 저장는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
