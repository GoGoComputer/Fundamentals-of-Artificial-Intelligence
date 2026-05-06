# 5.5 무기 3: Weight Decay (L2)

> **선행 학습:** Weight Decay는 **3장의 Ridge 정규화**와 같은 개념입니다.  
> [👉 정규화: Lasso, Ridge, ElasticNet](../03-ml-regression/05-regularization.md)  
> [👉 미분 입문](../../부록/수학/03-미분-입문.md) — 최적화 개념

---

## 머신러닝의 Ridge가 신경망에서는?

[3장 정규화](../03-ml-regression/05-regularization.md)에서 Ridge 회귀를 배웠죠. **가중치를 작게 만드는** 정규화 방식이었어요.

신경망에서도 같은 방식이 있어요. **Weight Decay** 라고 부릅니다. PyTorch 옵티마이저의 옵션 한 줄로 끝나요.

---

## 단 한 줄 추가

```python
# Weight Decay 없음
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Weight Decay 있음
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

`weight_decay=0.01` 이 한 부분이 전부예요. 진짜 한 줄.

---

## 어떻게 동작해요?

학습 시 손실 함수에 **가중치의 크기에 대한 페널티**를 더해요.

```
원래 손실: MSE
Weight Decay 추가: MSE + λ × Σwᵢ²
                          ↑
                       페널티
```

`λ` (람다)가 `weight_decay` 값이에요. 클수록 가중치를 더 작게 만들도록 강제합니다.

### 비유

학생이 답을 외워서 풀려고 해요. 그런데 선생님이 "답안지에 너무 많은 글자를 쓰면 감점이야" 라고 말합니다.

학생은 답을 외우면서도 **간단하게** 답해야 해요. 결과적으로 외우기보다는 핵심 패턴을 학습하게 됩니다.

Weight Decay가 신경망에 같은 효과를 줍니다. **단순한 모델로 학습하라고 강제**해요.

---

## weight_decay 값을 얼마로?

| weight_decay | 효과 |
|------------|------|
| 0 | 정규화 없음 |
| 1e-5 (0.00001) | 매우 약함 |
| 1e-4 (0.0001) | 약함 |
| 1e-3 (0.001) | 표준 |
| 1e-2 (0.01) | 강함 |
| 1e-1 (0.1) | 매우 강함 (학습 안 될 수도) |

**처음 시도: 1e-4 (0.0001)**

과적합이 심하면 키우고 (0.001 → 0.01), 학습이 안 되면 줄이세요.

---

## AdamW: 더 정확한 Weight Decay

`Adam`과 `AdamW`는 weight decay 처리 방식이 살짝 달라요.

| | Adam | AdamW |
|---|---|---|
| Weight Decay 처리 | 부정확 (L2 정규화로 변형됨) | 정확 (진짜 weight decay) |
| 일반화 성능 | 보통 | 더 좋음 |

**큰 모델이나 weight decay를 많이 쓸 땐 AdamW를 쓰세요.**

```python
# 권장
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

이게 현대 신경망의 표준 옵티마이저입니다 (Transformer 등).

---

## 가중치가 진짜 작아지나? 확인해 보기

```python
def get_weight_norm(model):
    """모든 가중치의 L2 norm 합."""
    norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            norm += p.norm(2).item() ** 2
    return norm ** 0.5


# 두 모델 비교
model_no_wd = MyModel()
optimizer_no = torch.optim.Adam(model_no_wd.parameters(), lr=0.001, weight_decay=0)

model_wd = MyModel()
optimizer_wd = torch.optim.Adam(model_wd.parameters(), lr=0.001, weight_decay=0.01)

# 둘 다 같은 데이터로 같은 step 만큼 학습 후
print(f"WD 없음: {get_weight_norm(model_no_wd):.4f}")
print(f"WD 0.01:  {get_weight_norm(model_wd):.4f}")
```

```
WD 없음: 145.32
WD 0.01:  87.46
```

가중치가 더 작아진 게 보여요.

---

## Weight Decay만 쓰지 마세요

Weight Decay 하나만으로는 충분하지 않아요. **Dropout, Early Stopping과 같이 쓰세요.**

세 가지를 조합해서 쓰는 게 표준 패턴입니다.

```python
# 표준 패턴
class RobustModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)    # ← Dropout
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        return self.fc3(x).squeeze()


model = RobustModel()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← Weight Decay
)

# 학습 시 Early Stopping 적용 ← 셋 다!
```

---

## 부분적으로 적용하기 (고급)

`weight_decay`를 모든 파라미터에 똑같이 적용하면 가끔 문제가 됩니다. 특히 **bias나 BatchNorm 파라미터**에는 weight decay를 적용 안 하는 게 좋아요.

```python
# bias와 BN을 빼고 weight decay 적용
decay = []
no_decay = []

for name, param in model.named_parameters():
    if 'bias' in name or 'bn' in name.lower() or 'norm' in name.lower():
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=0.001)
```

복잡해 보이지만, 큰 모델 학습할 때 표준이에요. 처음 배우실 땐 그냥 모두에 적용해도 큰 차이 없습니다.

---

## L1 vs L2 Weight Decay

PyTorch는 기본적으로 **L2 (제곱)** 만 지원해요. **L1**을 쓰고 싶으면 직접 추가해야 합니다.

### L2 (PyTorch 기본)

```python
# 자동
optimizer = torch.optim.AdamW(..., weight_decay=0.01)
```

페널티: `Σwᵢ²`

### L1 (직접 구현)

```python
loss_fn = nn.MSELoss()

# 학습 루프 안
loss = loss_fn(pred, y)
l1_penalty = sum(p.abs().sum() for p in model.parameters())
total_loss = loss + 0.001 * l1_penalty    # 0.001은 L1 강도
total_loss.backward()
```

**언제 어떤 걸?** 거의 항상 L2를 씁니다. L1은 특성 선택을 자동으로 하고 싶을 때 가끔.

---

## 정리

```python
# 1. 옵티마이저에 weight_decay 추가 (한 줄)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← 이거
)

# 2. 다른 정규화와 같이 쓰기
- Dropout (이전 글)
- BatchNorm
- Early Stopping
```

**한 줄 요약: "AdamW + weight_decay=0.01 부터 시작."**

---

## 자주 묻는 질문

> **Q. weight_decay와 Dropout 중 뭐가 더 강해요?**
>
> 둘이 다른 효과예요. 같이 쓰세요. weight_decay는 **모델을 작고 단순하게**, Dropout은 **앙상블 효과**.

> **Q. weight_decay 값이 너무 크면 어떻게 돼요?**
>
> Loss가 거의 안 떨어져요. 모델이 학습을 못 해요. 줄이세요.

> **Q. Adam과 AdamW 중 어느 걸 써요?**
>
> 모르겠으면 AdamW. 더 좋은 일반화. 단점은 거의 없어요.

➡️ 다음: [5.6 세 무기 다 같이 쓰기](06-combining-techniques.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **5.5 무기 3: Weight Decay (L2)** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **5.5 무기 3: Weight Decay (L2)**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "5.5 무기 3: Weight Decay (L2) 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **5.5 무기 3: Weight Decay (L2)**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

5.5 무기 3: Weight Decay (L2)는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
