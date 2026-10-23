# 실습 데이터 (data/)

실습 코드가 외부 서버 상태에 흔들리지 않도록, 작은 공개 데이터셋은 이 폴더에 함께 둡니다.

| 파일 | 내용 | 크기 | 출처 |
|------|------|------|------|
| `boston_housing.csv` | Boston Housing — 506개 동네 × 13개 특성 + 집값 중앙값(`MEDV`, 단위 $1,000) | 36KB | Harrison & Rubinfeld (1978), CMU StatLib 배포본과 동일한 값 |

## 불러오는 법

저장소를 내려받아 로컬에서 실행할 때는 파일을, Google Colab처럼 파일이 없는 환경에서는 GitHub 주소를 씁니다.
3·5·6장의 실습 코드는 모두 아래 패턴으로 데이터를 읽습니다.

```python
from pathlib import Path
import pandas as pd

BOSTON_URL = (
    "https://raw.githubusercontent.com/GoGoComputer/"
    "Fundamentals-of-Artificial-Intelligence/main/data/boston_housing.csv"
)
local_file = Path("data/boston_housing.csv")          # 저장소 루트에서 실행할 때
df = pd.read_csv(local_file) if local_file.exists() else pd.read_csv(BOSTON_URL)

X = df.drop(columns="MEDV")   # 13개 특성
y = df["MEDV"]                # 집값 중앙값
```

> ⚠️ Boston 데이터셋은 `B`(인종 구성) 변수처럼 사회적 편향이 담긴 열을 포함하고 있어 scikit-learn 1.2부터
> 기본 제공에서 제외되었습니다. 이 자료에서는 "회귀 알고리즘을 익히는 교재용 예제"로만 사용하며,
> 실제 서비스에 쓸 모델을 만들 때는 이런 변수를 넣지 않습니다. 자세한 이야기는 3장 2절과 7.8절을 보세요.

MNIST 손글씨 데이터는 7만 장(약 50MB)이라 저장소에 넣지 않고, 2장은 `sklearn.datasets.fetch_openml`,
4장은 `torchvision.datasets.MNIST` 로 내려받습니다. 네트워크가 막힌 환경에서는 2장 본문에 안내한
`sklearn.datasets.load_digits` (8×8 손글씨, 설치 시 함께 제공) 로 같은 실습을 할 수 있습니다.
