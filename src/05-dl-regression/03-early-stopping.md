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
