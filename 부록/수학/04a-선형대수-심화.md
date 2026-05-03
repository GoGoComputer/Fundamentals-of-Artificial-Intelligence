# 4a. 선형대수 심화 — 행렬식, 고유값, SVD, PCA

> [4장](04-선형대수-입문.md)에서 벡터와 행렬을 봤죠.
> 여기는 **행렬식, 고유값/고유벡터, SVD, PCA**를 채웁니다.
> ML의 핵심 도구 (PCA, 추천 시스템, Word2Vec 등) 의 수학적 기초입니다.

---

## 4a.1 행렬식 (Determinant)

### 직관

n×n **정사각행렬**에 대해 정의되는 **하나의 숫자**.

기하학적 의미: 행렬이 공간을 **얼마나 늘리거나 줄이는지**.

### 2×2 행렬

```
A = [a  b]
    [c  d]

det(A) = |A| = ad - bc
```

예시:
```
A = [3  1]    →   det = 3×4 - 1×2 = 10
    [2  4]
```

### 의미

```
det = 0  →  행렬이 차원을 축소시킴 (역행렬 없음, 정보 손실)
det ≠ 0  →  행렬이 가역 (역행렬 존재)
det < 0  →  반사 (방향 뒤집힘)
```

### Python

```python
import numpy as np
A = np.array([[3, 1], [2, 4]])
print(np.linalg.det(A))    # 10.0
```

### ML에서?

직접 사용은 적지만 알아두면:
- **역행렬 존재 여부**: det ≠ 0 일 때
- **가우시안 분포의 정규화 상수**에 등장
- **VAE, Normalizing Flows** 같은 생성 모델

---

## 4a.2 역행렬 (Inverse Matrix)

### 정의

`A · A⁻¹ = A⁻¹ · A = I` 인 행렬 `A⁻¹`.

수학적으로:
```
A⁻¹ = (1/det(A)) × adj(A)
```

(adj는 수반행렬. 깊이 안 들어갑니다.)

### 2×2 행렬의 역행렬

```
A = [a  b]            A⁻¹ = (1/(ad-bc)) × [d  -b]
    [c  d]                                  [-c  a]
```

### Python

```python
A_inv = np.linalg.inv(A)
print(A @ A_inv)    # ≈ 단위 행렬 I
```

### ML에서?

#### 선형 회귀의 정확한 해 (Closed-form)

```
w = (XᵀX)⁻¹ Xᵀ y
        ↑
     역행렬!
```

이게 최소제곱법(Least Squares)의 정확한 해. 다만:
- `XᵀX` 가 매우 클 수 있음 (계산 비용)
- 가끔 역행렬이 없음
- 그래서 보통은 **경사하강법** 으로 근사

#### 정규방정식 (Normal Equation)

작은 데이터에서 sklearn 의 `LinearRegression` 이 내부적으로 사용.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()    # 내부적으로 (XᵀX)⁻¹Xᵀy 계산
model.fit(X, y)
```

---

## 4a.3 선형 변환 (Linear Transformation)

### 직관

행렬을 곱하는 게 사실 **공간의 변환**이에요.

```
y = Ax
   ↑
   x를 y로 변환
```

### 종류

| 행렬 | 변환 |
|------|------|
| `[2, 0; 0, 2]` | 2배 확대 |
| `[1, 0; 0, -1]` | y축에 대칭 |
| `[cos θ, -sin θ; sin θ, cos θ]` | θ만큼 회전 |
| `[1, k; 0, 1]` | x축으로 기울임 (shear) |

### 시각화

```
원래 단위원:      변환 후:
   ●                   ●
  ╱│╲               ╱──────●
●─┼─●     →       ●─────────────●
  ╲│╱               ╲──────●
   ●                   ●
   
   (회전된 모양)
```

### 신경망 = 선형 변환의 연속

```
y = W₃ × σ(W₂ × σ(W₁ × x + b₁) + b₂) + b₃
```

각 `Wᵢ × ___` 가 선형 변환. `σ()`로 비선형성 추가.

만약 σ() 없으면 어떻게 될까요?

```
y = W₃W₂W₁ x + (b 항들)
  = W' x + b'
```

여러 층을 쌓아도 결국 한 층과 같음! 그래서 **활성화 함수가 필수**예요.

---

## 4a.4 고유값과 고유벡터 (Eigenvalues & Eigenvectors)

### 직관

`A v = λ v`

행렬 `A`로 변환했는데 **방향은 안 바뀌고 크기만 λ배** 되는 특별한 벡터 `v`.

- `v`: **고유벡터 (eigenvector)** — A의 특별한 방향
- `λ`: **고유값 (eigenvalue)** — 그 방향으로 얼마나 늘어나는지

### 예시

```
A = [3  1]   v = [1]   →  Av = [3+1] = [4] = 4 × [1]
    [1  3]       [1]            [1+3]   [4]      [1]

→ v = (1, 1) 은 고유벡터, 고유값 λ = 4
```

(다른 고유벡터는 `(1, -1)`, 고유값 `2`.)

### 어떻게 구해요?

`Av = λv` → `(A - λI)v = 0`

`v ≠ 0` 이려면 `det(A - λI) = 0`. 이게 **특성 방정식**.

#### 위 예시:

```
det([3-λ  1   ]) = 0
   ([1    3-λ ])

(3-λ)² - 1 = 0
λ² - 6λ + 8 = 0
(λ-2)(λ-4) = 0
λ = 2 또는 4
```

### Python

```python
A = np.array([[3, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)        # [4, 2]
print(eigenvectors)       # 각 열이 고유벡터
```

### ML에서?

#### 1. PCA (주성분 분석)

데이터의 **공분산 행렬의 고유벡터** = 주성분.
가장 큰 고유값에 대응하는 방향이 데이터가 **가장 많이 분산되는 방향**.

#### 2. 추천 시스템

행렬 분해 (Matrix Factorization). 사용자-아이템 행렬을 두 작은 행렬의 곱으로.

#### 3. 그래프 알고리즘

PageRank가 사실 행렬의 고유벡터 계산.

#### 4. 양자역학

(ML과 무관하지만, 양자컴퓨팅 분야에서 다 등장.)

---

## 4a.5 SVD (Singular Value Decomposition) — 특이값 분해

### 정의

**모든 행렬은 세 행렬의 곱으로 분해 가능.**

```
A = U Σ Vᵀ
```

- `A`: m×n 행렬 (어떤 행렬이든)
- `U`: m×m 직교행렬 (왼쪽 특이벡터)
- `Σ`: m×n 대각행렬 (특이값들)
- `V`: n×n 직교행렬 (오른쪽 특이벡터)

### 의미

```
A = U Σ Vᵀ

→ "A로 변환하는 일은 = V로 회전 → Σ로 스케일 → U로 회전"
```

### Python

```python
A = np.random.randn(5, 3)
U, S, Vt = np.linalg.svd(A)

print(U.shape)    # (5, 5)
print(S.shape)    # (3,)   ← 특이값들
print(Vt.shape)   # (3, 3)
```

### Truncated SVD — 차원 축소

특이값 중 큰 것만 남기고 작은 건 버림. 정보의 대부분은 큰 특이값들에 있어요.

```
A ≈ Uₖ Σₖ Vₖᵀ       (k < min(m, n))
```

### ML에서?

#### 1. 차원 축소

PCA가 사실은 데이터의 SVD 또는 공분산 행렬의 고유분해.

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X)    # 1000차원 → 50차원
```

#### 2. 추천 시스템

Netflix Prize의 우승 알고리즘이 SVD 기반.

```
사용자-영화 평점 행렬 R (1억×1만)
      ↓ SVD
R ≈ U Σ Vᵀ
      ↓
잠재 특성으로 평점 예측
```

#### 3. LSA (Latent Semantic Analysis)

문서-단어 행렬의 SVD → 의미 잠재 공간.

#### 4. 압축

이미지의 SVD로 근사 → 데이터 크기 줄임 (저장 압축).

---

## 4a.6 PCA (Principal Component Analysis) — 주성분 분석

선형대수 응용의 가장 유명한 예. **차원 축소**의 표준.

### 직관

고차원 데이터에서 **분산이 가장 큰 방향**들을 찾아 그걸로 표현.

```
2차원 데이터:           PC1 (가장 큰 분산 방향):
                            
    ●●●●                    ●●●●
   ●●●●●           →       ●●●●●         축 회전
    ●●●●                    ●●●●         후 PC1 축으로
                                          데이터 표현
```

### 알고리즘

1. 데이터를 평균 0으로 중심화
2. 공분산 행렬 계산: `C = (1/n) XᵀX`
3. 공분산 행렬의 **고유값과 고유벡터** 계산
4. 가장 큰 고유값에 대응하는 고유벡터 = PC1
5. 두 번째 = PC2, ...

### Python

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)    # 1000차원 → 2차원

print(pca.explained_variance_ratio_)
# [0.45, 0.23]   ← PC1이 분산의 45%, PC2가 23% 설명
```

### ML에서 어디?

- **차원 축소**: 모델 속도/메모리 개선
- **시각화**: 1000차원 데이터를 2D로 그리기
- **노이즈 제거**: 작은 PC들 버리기
- **얼굴 인식**: Eigenfaces (각 얼굴이 PC들의 조합)

### PCA의 한계

- **선형 변환만**: 비선형 구조 못 잡음 → t-SNE, UMAP 등
- **분산 ≠ 정보**: 가끔 작은 분산이 더 중요할 수도

---

## 4a.7 노름 (Norm) — 벡터의 크기

### L2 노름 (유클리드)

가장 자주 쓰는 벡터 크기. [1장](01-중학수학-복습.md)의 피타고라스 일반화.

```
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

### L1 노름 (맨해튼)

```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

### L∞ 노름

```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

### Python

```python
v = np.array([3, 4])
np.linalg.norm(v, ord=2)    # 5.0  (L2, 피타고라스)
np.linalg.norm(v, ord=1)    # 7.0  (L1)
np.linalg.norm(v, ord=np.inf)    # 4.0  (L∞)
```

### ML에서?

- **L2 정규화 (Ridge)**: `Σwᵢ²` = `||w||₂²`
- **L1 정규화 (Lasso)**: `Σ|wᵢ|` = `||w||₁`
- **거리 계산**: KNN의 거리 함수
- **Gradient Clipping**: `||∇L||` 가 클 때 잘라냄

---

## 4a.8 직교성 (Orthogonality)

### 두 벡터가 직교 (수직)

```
u · v = 0    (내적이 0)
```

### 정규직교 (Orthonormal)

직교이면서 각자 크기가 1.

```
u · v = 0 (직교)
||u|| = ||v|| = 1 (단위 벡터)
```

### 직교 행렬 (Orthogonal Matrix)

각 열이 정규직교인 행렬.

```
QᵀQ = QQᵀ = I
Q⁻¹ = Qᵀ      (역행렬 = 전치)
```

### ML에서?

- **PCA의 주성분들** = 서로 직교 (직교 변환)
- **회전 행렬**은 직교 행렬
- **신경망 가중치 초기화**: 직교 초기화로 학습 안정화
- **SVD의 U, V** 는 직교 행렬

---

## 4a.9 정리

```python
import numpy as np

# 행렬식
np.linalg.det(A)

# 역행렬
np.linalg.inv(A)

# 고유값, 고유벡터
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)

# 노름
np.linalg.norm(v, ord=2)    # L2
np.linalg.norm(v, ord=1)    # L1

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(X)
```

### ML 식 다시 읽기

```
A v = λ v
```
"행렬 A에 벡터 v를 곱한 결과 = 람다 곱하기 v." → 고유벡터/고유값 정의.

```
A = U Σ Vᵀ
```
"행렬 A = (직교행렬 U) × (대각행렬 Σ) × (직교행렬 V의 전치)." → SVD.

```
w = (XᵀX)⁻¹ Xᵀy
```
"가중치 = (X 전치 X 의 역행렬) × X 전치 × y." → 선형 회귀의 정확한 해.

```
||w||₂² = Σ wᵢ²
```
"w의 L2 노름의 제곱 = 모든 wᵢ 제곱의 합." → L2 정규화 페널티.

---

## 자가 진단

1. `[2, 1; 3, 4]` 의 행렬식은?
2. `[3, 1; 1, 3]` 의 고유값은?
3. SVD가 차원 축소에 어떻게 쓰여요?
4. PCA의 주성분은 무엇의 고유벡터?
5. 직교 행렬의 역행렬을 빠르게 구하려면?

<details>
<summary>정답</summary>

1. `2×4 - 1×3 = 5`
2. `λ = 2 와 4` (위 본문 참조)
3. 작은 특이값들 무시하고 큰 것만 남김 → 데이터를 더 작은 차원으로 표현
4. **공분산 행렬**의 고유벡터
5. 전치 (`Q⁻¹ = Qᵀ`)

</details>

➡️ 다음: [5. 확률통계 입문](05-확률통계-입문.md) (또는 [5a. 확률통계 심화](05a-확률통계-심화.md))
