"""6장 3절 실습: FastAPI로 모델 서빙

01_모델_저장.py 를 먼저 실행해서 모델을 저장해 두세요.
이 파일을 실행하시려면:
    pip install fastapi uvicorn pydantic joblib

실행:
    uvicorn 02_FastAPI_서빙:app --reload --port 8000

또는:
    python 02_FastAPI_서빙.py

실행 후 http://localhost:8000/docs 에서 인터페이스 확인.
"""

import logging
import time
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ============================================================
# 로깅
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================
# 모델 불러오기 (앱 시작 시 한 번)
# ============================================================
MODEL_DIR = Path('models/boston_v1.0.0')
PIPELINE_PATH = MODEL_DIR / 'pipeline.pkl'
METADATA_PATH = MODEL_DIR / 'metadata.json'

if not PIPELINE_PATH.exists():
    raise RuntimeError(
        f"모델 파일이 없습니다: {PIPELINE_PATH}\n"
        f"01_모델_저장.py를 먼저 실행해 주세요."
    )

pipeline = joblib.load(PIPELINE_PATH)

with open(METADATA_PATH, encoding='utf-8') as f:
    metadata = json.load(f)

MODEL_VERSION = metadata['version']
FEATURE_NAMES = metadata['data_info']['features']

logger.info(f"모델 불러옴: v{MODEL_VERSION}")


# ============================================================
# 입력/출력 스키마
# ============================================================
class HouseFeatures(BaseModel):
    CRIM: float = Field(..., description="범죄율", ge=0)
    ZN: float = Field(..., description="주거지 비율", ge=0)
    INDUS: float = Field(..., description="비상업 토지 비율", ge=0)
    CHAS: int = Field(..., ge=0, le=1, description="강 옆 (0/1)")
    NOX: float = Field(..., description="오염", ge=0, le=1)
    RM: float = Field(..., description="평균 방 개수", gt=0)
    AGE: float = Field(..., description="40년 이상 비율", ge=0, le=100)
    DIS: float = Field(..., description="도심까지 거리", ge=0)
    RAD: float = Field(..., description="고속도로 접근성")
    TAX: float = Field(..., description="재산세율", ge=0)
    PTRATIO: float = Field(..., description="학생-교사 비율", gt=0)
    B: float = Field(..., description="B 변수", ge=0)
    LSTAT: float = Field(..., description="저소득층 비율", ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31,
                "CHAS": 0, "NOX": 0.538, "RM": 6.575,
                "AGE": 65.2, "DIS": 4.0900, "RAD": 1.0,
                "TAX": 296.0, "PTRATIO": 15.3, "B": 396.90,
                "LSTAT": 4.98,
            },
        }


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="예측 집값 ($1000)")
    model_version: str
    inference_time_ms: float


class BatchRequest(BaseModel):
    samples: List[HouseFeatures]


class BatchResponse(BaseModel):
    predictions: List[float]
    model_version: str
    n_samples: int
    inference_time_ms: float


# ============================================================
# FastAPI 앱
# ============================================================
app = FastAPI(
    title="Boston House Price Predictor",
    description="보스턴 지역 집값 예측 API",
    version=MODEL_VERSION,
)


# ============================================================
# 미들웨어: 모든 요청 로깅
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} -> "
        f"{response.status_code} ({duration_ms:.2f}ms)"
    )
    return response


# ============================================================
# 엔드포인트들
# ============================================================
@app.get("/")
def root():
    return {
        "service": "Boston House Price Predictor",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch",
            "health": "GET /health",
            "info": "GET /info",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    """헬스 체크."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/info")
def info():
    """모델 정보."""
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """단일 예측."""
    start = time.time()

    try:
        # dict → DataFrame (sklearn pipeline은 DataFrame 받음)
        input_df = pd.DataFrame([features.model_dump()])

        # 컬럼 순서 보장
        input_df = input_df[FEATURE_NAMES]

        # 예측
        prediction = float(pipeline.predict(input_df)[0])

        # 음수 방지
        prediction = max(0.0, prediction)

        duration_ms = (time.time() - start) * 1000

        return PredictionResponse(
            predicted_price=prediction,
            model_version=MODEL_VERSION,
            inference_time_ms=duration_ms,
        )

    except Exception as e:
        logger.error(f"예측 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """배치 예측."""
    start = time.time()

    if not request.samples:
        raise HTTPException(status_code=400, detail="At least one sample required")

    if len(request.samples) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 samples per request")

    try:
        input_df = pd.DataFrame([s.model_dump() for s in request.samples])
        input_df = input_df[FEATURE_NAMES]

        predictions = pipeline.predict(input_df)
        predictions = np.maximum(0, predictions).tolist()

        duration_ms = (time.time() - start) * 1000

        return BatchResponse(
            predictions=predictions,
            model_version=MODEL_VERSION,
            n_samples=len(predictions),
            inference_time_ms=duration_ms,
        )

    except Exception as e:
        logger.error(f"배치 예측 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ============================================================
# 에러 핸들러
# ============================================================
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("API 시작 중...")
    print("문서: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
