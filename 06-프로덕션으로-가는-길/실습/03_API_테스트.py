"""6장 3절 실습: API 테스트

먼저 02_FastAPI_서빙.py가 실행 중이어야 합니다.
다른 터미널에서 이 파일을 실행하세요.

실행 전에:
    pip install requests
"""

import requests
import time
import json


BASE_URL = "http://localhost:8000"


# ============================================================
# 1. Health Check
# ============================================================
print("=" * 60)
print("1. Health Check")
print("=" * 60)

resp = requests.get(f"{BASE_URL}/health")
print(f"Status: {resp.status_code}")
print(f"Body: {resp.json()}")


# ============================================================
# 2. 서비스 정보
# ============================================================
print("\n" + "=" * 60)
print("2. Root")
print("=" * 60)

resp = requests.get(f"{BASE_URL}/")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


# ============================================================
# 3. 모델 정보
# ============================================================
print("\n" + "=" * 60)
print("3. Model Info")
print("=" * 60)

resp = requests.get(f"{BASE_URL}/info")
data = resp.json()
print(f"버전: {data['version']}")
print(f"모델 종류: {data['model_info']['type']}")
print(f"성능: {data['metrics']}")


# ============================================================
# 4. 단일 예측
# ============================================================
print("\n" + "=" * 60)
print("4. 단일 예측")
print("=" * 60)

example_house = {
    "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31,
    "CHAS": 0, "NOX": 0.538, "RM": 6.575,
    "AGE": 65.2, "DIS": 4.0900, "RAD": 1.0,
    "TAX": 296.0, "PTRATIO": 15.3, "B": 396.90,
    "LSTAT": 4.98,
}

resp = requests.post(f"{BASE_URL}/predict", json=example_house)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    result = resp.json()
    print(f"예측 집값: ${result['predicted_price']*1000:,.0f}")
    print(f"추론 시간: {result['inference_time_ms']:.2f}ms")
    print(f"모델 버전: {result['model_version']}")


# ============================================================
# 5. 배치 예측
# ============================================================
print("\n" + "=" * 60)
print("5. 배치 예측 (5개 한 번에)")
print("=" * 60)

batch_request = {
    "samples": [
        # 1번 집 (저렴)
        {"CRIM": 88.0, "ZN": 0, "INDUS": 18, "CHAS": 0, "NOX": 0.7,
         "RM": 4.5, "AGE": 95, "DIS": 1.5, "RAD": 24, "TAX": 666,
         "PTRATIO": 20, "B": 100, "LSTAT": 35},
        # 2번 집 (보통)
        {"CRIM": 0.5, "ZN": 0, "INDUS": 8, "CHAS": 0, "NOX": 0.5,
         "RM": 6.0, "AGE": 50, "DIS": 5.0, "RAD": 4, "TAX": 300,
         "PTRATIO": 16, "B": 380, "LSTAT": 12},
        # 3번 집 (좋음)
        example_house,
        # 4번 집 (고급)
        {"CRIM": 0.01, "ZN": 80, "INDUS": 1, "CHAS": 1, "NOX": 0.4,
         "RM": 8.5, "AGE": 5, "DIS": 8.0, "RAD": 1, "TAX": 250,
         "PTRATIO": 14, "B": 396, "LSTAT": 2},
        # 5번 집 (또 보통)
        {"CRIM": 1.0, "ZN": 0, "INDUS": 8, "CHAS": 0, "NOX": 0.55,
         "RM": 6.2, "AGE": 60, "DIS": 4.0, "RAD": 4, "TAX": 320,
         "PTRATIO": 17, "B": 390, "LSTAT": 14},
    ]
}

resp = requests.post(f"{BASE_URL}/predict_batch", json=batch_request)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    result = resp.json()
    print(f"\n예측 결과 ({result['n_samples']}건):")
    for i, pred in enumerate(result['predictions']):
        print(f"  {i+1}번 집: ${pred*1000:,.0f}")
    print(f"\n총 추론 시간: {result['inference_time_ms']:.2f}ms")
    print(f"건당 평균: {result['inference_time_ms']/result['n_samples']:.2f}ms")


# ============================================================
# 6. 잘못된 입력
# ============================================================
print("\n" + "=" * 60)
print("6. 잘못된 입력 (검증 테스트)")
print("=" * 60)

invalid_request = {
    "CRIM": -1,    # 음수 (안 됨!)
    "ZN": 18.0, "INDUS": 2.31,
    "CHAS": 5,    # 0 또는 1만 가능
    "NOX": 0.538, "RM": 6.575,
    "AGE": 65.2, "DIS": 4.0900, "RAD": 1.0,
    "TAX": 296.0, "PTRATIO": 15.3, "B": 396.90,
    "LSTAT": 4.98,
}

resp = requests.post(f"{BASE_URL}/predict", json=invalid_request)
print(f"Status: {resp.status_code} (422가 정상)")
if resp.status_code == 422:
    errors = resp.json()['detail']
    print(f"\n검증 실패한 필드:")
    for err in errors:
        print(f"  - {err['loc'][-1]}: {err['msg']}")


# ============================================================
# 7. 부하 테스트 (간단)
# ============================================================
print("\n" + "=" * 60)
print("7. 100번 호출 (부하 테스트)")
print("=" * 60)

n_requests = 100
start = time.time()
success_count = 0

for _ in range(n_requests):
    resp = requests.post(f"{BASE_URL}/predict", json=example_house)
    if resp.status_code == 200:
        success_count += 1

duration = time.time() - start
print(f"\n결과:")
print(f"  성공: {success_count}/{n_requests}")
print(f"  총 시간: {duration:.2f}초")
print(f"  평균 응답: {duration/n_requests*1000:.2f}ms")
print(f"  처리량: {n_requests/duration:.1f} req/s")


print("\n테스트 끝!")
