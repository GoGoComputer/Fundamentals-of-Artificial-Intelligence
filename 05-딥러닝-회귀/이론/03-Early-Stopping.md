# 5.3 무기 1: Early Stopping

## 단순하지만 강력한

Early Stopping은 **이름 그대로** 학습을 일찍 멈추는 거예요. 단순한 아이디어지만, 효과가 매우 좋습니다.

---

## 왜 일찍 멈추는 게 좋아요?

학습이 진행될수록 어떤 일이 일어나는지 봅시다.

```
Epoch  Train Loss  Test Loss
  1      1.5         1.4       ← 둘 다 큼 (학습 시작)
  10     0.5         0.6       ← 둘 다 줄어듦
  20     0.3         0.4       ← 더 줄어듦
  30     0.2         0.35      ← Test Loss는 천천히
  40     0.1         0.4       ← Test Loss가 다시 올라감!
  50     0.05        0.5       ← 명확한 과적합
  60     0.02        0.7       ← 끔찍한 과적합
```

30 epoch까지가 좋아요. 그 이후엔 과적합이 시작돼요.

**Early Stopping의 아이디어:** "Test Loss가 안 좋아지면 멈춰라."

---

## 구현 방법

### 핵심 아이디어

```
patience = 5    # 몇 번까지 참을까

best_val_loss = ∞
patience_counter = 0

for epoch:
    train()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:    # 개선됨
        best_val_loss = val_loss
        patience_counter = 0
        save_model()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break    # 멈춤!
```

### 진짜 코드

```python
import torch

def train_with_early_stopping(
    model, train_loader, val_loader,
    n_epochs=200, patience=10, lr=0.001,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_history = []
    val_history = []
    
    for epoch in range(n_epochs):
        # === 학습 ===
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # === 검증 ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(val_loader)
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        # === Early Stopping 로직 ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best.pth')
            print(f"  → 베스트 모델 저장 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → 개선 없음 ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # 베스트 모델 불러오기
    model.load_state_dict(torch.load('best.pth'))
    return model, train_history, val_history
```

### 사용

```python
model = MyRegressor()
model, train_hist, val_hist = train_with_early_stopping(
    model, train_loader, val_loader,
    n_epochs=200, patience=10,
)
```

학습이 자동으로 적절한 시점에 멈춰요. 마지막엔 **베스트 모델이 자동으로 불러와지죠.**

---

## patience를 얼마로 해야 해요?

`patience`는 "이만큼 참아 보고 안 개선되면 포기" 라는 값이에요.

| patience | 효과 |
|---------|------|
| 작음 (3-5) | 빨리 멈춤. 일찍 멈출 수도. |
| 중간 (10-20) | 표준적. 대부분 잘 동작. |
| 큼 (50+) | 천천히 결정. 시간 많이 걸림. |

데이터 크기와 학습 안정성에 따라 다르지만, **10~20** 정도가 좋은 시작점입니다.

---

## min_delta 추가

"개선됐다"의 기준을 좀 더 엄격하게 줄 수도 있어요.

```python
min_delta = 0.001    # 0.001 이상 개선돼야 진짜 개선

if val_loss < best_val_loss - min_delta:    # 진짜로 개선됨
    best_val_loss = val_loss
    patience_counter = 0
    ...
```

소수점 셋째 자리까지 미세하게 흔들리는 건 무시하고, 의미 있는 개선만 인정.

---

## 클래스로 만들어 두기

매번 위 로직을 다시 짤 필요 없어요. 한 번 만들어 두고 재사용하세요.

```python
class EarlyStopping:
    """Early Stopping 도구 클래스."""
    
    def __init__(self, patience=10, min_delta=0.0, save_path='best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        
        self.best_score = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_score - self.min_delta:
            # 개선됨
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True    # 개선됨을 알림
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def load_best(self, model):
        """베스트 모델로 복원."""
        model.load_state_dict(torch.load(self.save_path))
        return model


# 사용
early_stopping = EarlyStopping(patience=10)

for epoch in range(n_epochs):
    train_one_epoch()
    val_loss = evaluate()
    
    improved = early_stopping(val_loss, model)
    print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}", 
          "✅ 개선" if improved else f"❌ {early_stopping.counter}/{early_stopping.patience}")
    
    if early_stopping.early_stop:
        print("Early Stopping!")
        break

model = early_stopping.load_best(model)
```

---

## 시각화: Early Stopping이 일어난 시점

학습 후 시각화하면 효과가 한눈에 보여요.

```python
import matplotlib.pyplot as plt

best_epoch = train_hist.index(min(val_hist))    # 베스트 시점

plt.figure(figsize=(10, 5))
plt.plot(train_hist, label='Train Loss')
plt.plot(val_hist, label='Val Loss')
plt.axvline(best_epoch, color='red', linestyle='--', 
            label=f'Best Epoch ({best_epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping이 일어난 시점')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

빨간 점선이 베스트 시점이에요. 이 시점의 모델이 저장됐고, 학습이 그 이후 patience만큼 더 진행되다 멈춘 거예요.

---

## 함정과 주의사항

### 함정 1: patience를 너무 작게

```
patience = 1   ← 위험
```

검증 loss는 자연스럽게 흔들려요. 한 번 올라갔다고 바로 멈추면, **진짜 베스트를 놓칠 수 있어요.**

### 함정 2: Validation Set이 너무 작음

```
val_set 10개   ← 위험
```

작은 검증 셋은 점수가 매번 크게 흔들려요. 신뢰할 수 없어요. **최소 100개, 가능하면 1000개 이상.**

### 함정 3: Train과 Val에 겹침

학습 데이터를 검증 셋에 일부 넣으면, 검증 점수가 **실제보다 좋게** 나와요. **확실하게 분리**해야 해요.

```python
# ❌ 잘못
for epoch in range(...):
    val_idx = random.sample(range(len(train)), 100)
    val_set = train[val_idx]    # train에서 매번 무작위 → 겹칠 수 있음

# ✅ 올바름
train, val = random_split(full_train, [80%, 20%])    # 한 번 확정
```

### 함정 4: 모델 저장 안 함

```python
# ❌ 베스트 모델을 저장 안 하면 의미가 적음
if val_loss < best:
    best = val_loss
    counter = 0
    # save 안 함 ←
```

학습 끝나면 마지막 epoch의 모델이 남아요. 그게 베스트가 아닐 수 있어요. **꼭 저장하세요.**

---

## 다른 변형들

### 메트릭 기반 (loss 대신 accuracy 등)

```python
# 분류라면
class EarlyStoppingMaximize:
    """val_acc처럼 클수록 좋은 메트릭에 사용."""
    def __init__(self, patience=10):
        self.best = -float('inf')
        self.counter = 0
        self.patience = patience
        self.early_stop = False
    
    def __call__(self, score, model):
        if score > self.best:
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), 'best.pth')
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
```

### 학습률 줄이기 (Reduce LR on Plateau)

학습이 정체되면 멈추는 대신, 학습률을 줄여서 다시 시도.

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

for epoch in range(n_epochs):
    train_one_epoch()
    val_loss = evaluate()
    scheduler.step(val_loss)    # patience 동안 안 좋아지면 lr 절반으로
```

이걸 Early Stopping과 함께 쓰면 좋아요. **먼저 lr을 줄이다가, 그래도 안 되면 멈춤.**

---

## 정리

```python
# 가장 단순한 패턴
best_val = float('inf')
patience = 10
counter = 0

for epoch in range(n_epochs):
    train()
    val_loss = evaluate()
    
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        counter += 1
        if counter >= patience:
            break

# 베스트 불러오기
model.load_state_dict(torch.load('best.pth'))
```

**한 줄 요약: "검증 loss가 patience만큼 안 좋아지면 멈추기. 베스트는 저장."**

---

## 자주 묻는 질문

> **Q. Early Stopping만으로 충분한가요?**
>
> 보통 부족해요. Dropout, Weight Decay와 같이 쓰는 게 표준이에요. 다음 글들에서 다룹니다.

> **Q. patience를 너무 크게 하면 안 좋아요?**
>
> 시간이 오래 걸리지만, 학습 결과 자체는 더 좋을 수 있어요. 시간 여유 있으면 크게 해도 OK.

> **Q. 매 epoch가 아니라 매 N step마다 검증해도 되나요?**
>
> 됩니다. 큰 데이터에서는 한 epoch이 길어서 그렇게 해요. 다만 코드가 복잡해지므로 처음에는 epoch 단위가 편해요.

➡️ 다음: [5.4 무기 2: Dropout](04-Dropout.md)


<!-- THEORY_EXPANDED_2026_05 -->

---

## 이해를 더 단단하게 만드는 보강 설명

이 단원은 처음 읽을 때는 "아, 이런 개념이 있구나" 정도로 넘어가기 쉽지만, 실제로는 작은 차이가 나중에 큰 성능 차이로 이어집니다. 특히 **5.3 무기 1: Early Stopping** 같은 주제는 한 번에 완벽히 이해하려고 하기보다, 다음 세 가지를 반복하면서 이해를 깊게 만드는 편이 훨씬 효과적입니다.

첫째, 용어를 정의로 외우지 말고 상황으로 연결해 보세요. 예를 들어 "왜 이 개념이 필요한가"를 스스로 설명할 수 있어야 다음 단원에서 응용이 됩니다. 둘째, 코드가 나오면 결과를 먼저 예측한 뒤 실행해 보세요. 예측과 결과가 다를 때가 진짜 학습이 일어나는 순간입니다. 셋째, 같은 코드를 숫자나 파라미터만 바꿔 여러 번 돌려 보세요. 머신러닝과 딥러닝은 정답을 암기하는 과목이 아니라 **변화를 읽는 과목**이기 때문입니다.

또 하나 중요한 점은, 이 단원의 내용이 단독으로 존재하지 않는다는 사실입니다. 실제 프로젝트에서는 데이터 준비, 모델링, 평가, 배포가 끊기지 않고 이어집니다. 그래서 지금 배우는 **5.3 무기 1: Early Stopping**도 나중에 다른 단계와 반드시 연결됩니다. 이 연결을 의식하면서 읽으면, 단편 지식이 아니라 시스템 관점으로 이해하게 됩니다.

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
예: "5.3 무기 1: Early Stopping 관점에서 입력으로부터 어떤 출력을 만들 것인가?"

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

이 3개를 직접 해 보면 **5.3 무기 1: Early Stopping**의 이해도는 읽기만 했을 때보다 훨씬 빠르게 올라갑니다.

---

## 한 번 더 정리

5.3 무기 1: Early Stopping는 단순히 시험용 개념이 아니라, 실제 문제를 풀 때 반복해서 쓰게 되는 실전 도구입니다. 처음에는 낯설 수 있지만, 작은 실험을 반복하면 감각이 생기고, 감각이 생기면 코드와 지표를 읽는 속도가 급격히 빨라집니다. 지금 단계에서 중요한 것은 완벽한 암기가 아니라, **작게 실행하고, 결과를 읽고, 다시 수정하는 루프**를 몸에 익히는 것입니다. 이 루프를 익히면 다음 장으로 넘어갈수록 학습 속도가 오히려 빨라집니다.
