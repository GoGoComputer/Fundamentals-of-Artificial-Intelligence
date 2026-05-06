# 6.3 FastAPI로 모델 서빙

## 모델을 API로 만든다는 것

지금 모델은 **노트북에서만** 동작해요. 다른 사람이 쓰려면?

답은 **API**예요. **A**pplication **P**rogramming **I**nterface — 다른 프로그램이 우리 모델을 호출할 수 있게 하는 인터페이스.

```
사용자의 앱
     ↓
   HTTP 요청
     ↓
  우리 API 서버 ← 여기에 모델이 있음
     ↓
  HTTP 응답 (예측 결과)
     ↓
사용자의 앱
```

이 API 서버를 가장 쉽게 만드는 도구가 **FastAPI**예요.

---

## FastAPI가 뭐예요?

Python으로 API 서버를 만드는 라이브러리예요. 특징:

- ✅ **빠름** — 가장 빠른 Python 웹 프레임워크 중 하나
- ✅ **쉬움** — 짧은 코드로 가능
- ✅ **자동 문서화** — Swagger UI 자동 제공
- ✅ **타입 힌트 활용** — 자동 검증

ML API는 거의 모두 FastAPI로 만들어요.

### 설치

```bash
pip install fastapi uvicorn pydantic
```

(`uvicorn`은 FastAPI 앱을 실행하는 서버, `pydantic`은 데이터 검증)

---

## 가장 단순한 FastAPI 앱

```python
# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, ML World!"}
```

실행:

```bash
uvicorn app:app --reload
```

브라우저에서 `http://localhost:8000` 으로 가시면 JSON 응답이 보여요.

`/docs` 로 가시면 자동 생성된 API 문서가 있어요. 진짜 멋져요.

---

## 모델 서빙 API

이제 모델을 붙여 봅시다. Boston 집값 예측 모델을 가정.

### 1단계: 입력 / 출력 스키마 정의

```python
from pydantic import BaseModel, Field
from typing import List


class HouseFeatures(BaseModel):
    """입력: 집 정보."""
    crim: float = Field(..., description="범죄율")
    zn: float = Field(..., description="주거 비율")
    indus: float = Field(..., description="비상업 토지 비율")
    chas: int = Field(..., ge=0, le=1, description="강 옆이면 1")
    nox: float = Field(..., description="오염")
    rm: float = Field(..., description="방 개수")
    age: float = Field(..., description="40년 이상 비율")
    dis: float = Field(..., description="도심까지 거리")
    rad: float = Field(..., description="고속도로 접근성")
    tax: float = Field(..., description="재산세율")
    ptratio: float = Field(..., description="학생-교사 비율")
    b: float = Field(..., description="B 변수")
    lstat: float = Field(..., description="저소득층 비율")
    
    # 사용 예시 (자동 문서에 표시됨)
    class Config:
        json_schema_extra = {
            "example": {
                "crim": 0.00632,
                "zn": 18.0,
                "indus": 2.31,
                "chas": 0,
                "nox": 0.538,
                "rm": 6.575,
                "age": 65.2,
                "dis": 4.0900,
                "rad": 1.0,
                "tax": 296.0,
                "ptratio": 15.3,
                "b": 396.90,
                "lstat": 4.98,
            },
        }


class PredictionResponse(BaseModel):
    """출력: 예측 결과."""
    predicted_price: float = Field(..., description="예측 집값 (USD * 1000)")
    model_version: str
```

`pydantic`이 입력을 자동으로 검증해 줍니다. 잘못된 입력이면 알아서 에러 응답.

### 2단계: 모델 불러오기

```python
import joblib

# 앱 시작 시 한 번 불러옴
pipeline = joblib.load('models/boston_v1.0.0/pipeline.pkl')
MODEL_VERSION = '1.0.0'
```

### 3단계: 엔드포인트 만들기

```python
from fastapi import FastAPI, HTTPException
import numpy as np

app = FastAPI(
    title="Boston House Price Predictor",
    description="보스턴 지역 집값 예측 API",
    version=MODEL_VERSION,
)


@app.get("/")
def root():
    return {
        "service": "Boston House Price Predictor",
        "version": MODEL_VERSION,
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
    """헬스 체크 (모니터링용)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """집값 예측."""
    try:
        # dict → array
        feature_array = np.array([[
            features.crim, features.zn, features.indus,
            features.chas, features.nox, features.rm,
            features.age, features.dis, features.rad,
            features.tax, features.ptratio, features.b,
            features.lstat,
        ]])
        
        # 예측
        prediction = pipeline.predict(feature_array)[0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4단계: 실행

```bash
uvicorn app:app --reload --port 8000
```

이제 `http://localhost:8000/docs`로 가시면 **자동 생성된 인터페이스**에서 직접 API를 테스트할 수 있어요.

---

## API 호출 예시

### Python에서

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "crim": 0.00632, "zn": 18.0, "indus": 2.31,
        "chas": 0, "nox": 0.538, "rm": 6.575,
        "age": 65.2, "dis": 4.0900, "rad": 1.0,
        "tax": 296.0, "ptratio": 15.3, "b": 396.90,
        "lstat": 4.98,
    },
)
print(response.json())
# {"predicted_price": 23.45, "model_version": "1.0.0"}
```

### curl에서

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632, "zn": 18.0, "indus": 2.31,
    "chas": 0, "nox": 0.538, "rm": 6.575,
    "age": 65.2, "dis": 4.09, "rad": 1.0,
    "tax": 296.0, "ptratio": 15.3, "b": 396.90,
    "lstat": 4.98
  }'
```

---

## 배치 예측 (여러 개 한꺼번에)

한 번에 여러 집을 예측하고 싶으면:

```python
class BatchRequest(BaseModel):
    samples: List[HouseFeatures]


class BatchResponse(BaseModel):
    predictions: List[float]
    model_version: str


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    feature_arrays = np.array([[
        s.crim, s.zn, s.indus, s.chas, s.nox, s.rm,
        s.age, s.dis, s.rad, s.tax, s.ptratio, s.b, s.lstat,
    ] for s in request.samples])
    
    predictions = pipeline.predict(feature_arrays)
    
    return BatchResponse(
        predictions=predictions.tolist(),
        model_version=MODEL_VERSION,
    )
```

배치 예측이 한 개씩 100번 부르는 것보다 훨씬 빨라요.

---

## 로깅 추가

운영에서는 모든 요청을 로그로 남기는 게 좋아요.

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({duration:.2f}ms)"
    )
    return response
```

---

## 에러 처리

다양한 에러 상황에 우아하게 대응.

```python
from fastapi import HTTPException
from pydantic import ValidationError


@app.exception_handler(ValidationError)
async def validation_handler(request, exc):
    """입력 검증 실패."""
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": exc.errors()},
    )


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        prediction = pipeline.predict(features_to_array(features))[0]
        
        # 비즈니스 검증
        if prediction < 0:
            logger.warning(f"Negative prediction: {prediction}")
            prediction = 0.0
        
        return {"predicted_price": float(prediction)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")
```

---

## 인증 (간단)

API에 누구나 접근하면 안 되는 경우.

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(features: HouseFeatures):
    # ...
```

이제 `X-API-Key` 헤더 없으면 403 에러.

---

## 비동기 처리 (성능 향상)

큰 모델이라 추론이 느리면, 비동기로 처리:

```python
import asyncio


@app.post("/predict")
async def predict(features: HouseFeatures):
    # CPU 바운드 작업은 별도 스레드/프로세스로
    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(
        None, pipeline.predict, features_to_array(features),
    )
    return {"predicted_price": float(prediction[0])}
```

---

## 전체 코드를 한 곳에

```python
# app.py
import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 불러오기
pipeline = joblib.load('models/boston_v1.0.0/pipeline.pkl')
MODEL_VERSION = '1.0.0'

# FastAPI 앱
app = FastAPI(
    title="Boston House Price API",
    version=MODEL_VERSION,
)


class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int = Field(..., ge=0, le=1)
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float


class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        feature_array = np.array([[
            features.crim, features.zn, features.indus,
            features.chas, features.nox, features.rm,
            features.age, features.dis, features.rad,
            features.tax, features.ptratio, features.b,
            features.lstat,
        ]])
        
        prediction = float(pipeline.predict(feature_array)[0])
        
        # 음수 방지
        prediction = max(0, prediction)
        
        logger.info(f"Prediction: {prediction:.2f}")
        
        return PredictionResponse(
            predicted_price=prediction,
            model_version=MODEL_VERSION,
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

이걸 `python app.py` 또는 `uvicorn app:app`로 실행.

---

## 정리

```python
# FastAPI 핵심 패턴
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    feature: float

@app.post("/predict")
def predict(input: Input):
    result = model.predict([[input.feature]])
    return {"prediction": float(result[0])}

# 실행: uvicorn app:app
```

**핵심:**
- 입력은 pydantic BaseModel로 (자동 검증)
- 모델은 앱 시작 시 한 번 불러오기
- `/health` 엔드포인트로 모니터링
- 에러는 우아하게 처리
- 자동 문서: `/docs`

➡️ 다음: [6.4 Docker로 패키징](04-Docker.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **6.3 FastAPI로 모델 서빙** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **6.3 FastAPI로 모델 서빙**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "6.3 FastAPI로 모델 서빙 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **6.3 FastAPI로 모델 서빙**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

6.3 FastAPI로 모델 서빙는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
