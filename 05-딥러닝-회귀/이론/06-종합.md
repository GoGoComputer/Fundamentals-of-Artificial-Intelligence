# 5.6 세 무기 다 같이 쓰기

## 종합 코드

지금까지 배운 세 가지 무기를 다 적용한 신경망 회귀 코드입니다.
**이게 현업 표준 패턴**이에요. 그대로 가져다 쓰셔도 됩니다.

---

## 전체 코드

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# 시드 고정
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. 데이터 준비
# ============================================================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


# 3 way split: train, val, test
X_temp, X_test, y_temp, y_test = train_test_split(
    data, target, test_size=0.15, random_state=SEED,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=SEED,
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# 정규화 (X와 y 둘 다)
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))


def to_tensor(X, y):
    X_s = scaler_X.transform(X)
    y_s = scaler_y.transform(y.reshape(-1, 1)).flatten()
    return (
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.float32),
    )


X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)


# DataLoader
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=32, shuffle=False,
)


# ============================================================
# 2. 모델 (Dropout + BatchNorm 포함)
# ============================================================
class RobustRegressor(nn.Module):
    def __init__(self, input_dim=13, hidden=64, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.dropout2 = nn.Dropout(dropout_p)
        
        self.fc3 = nn.Linear(hidden // 2, 1)
    
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobustRegressor(input_dim=X_train.shape[1]).to(device)
print(model)


# ============================================================
# 3. 손실 + 옵티마이저 (AdamW + Weight Decay)
# ============================================================
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← Weight Decay
)


# ============================================================
# 4. Early Stopping 클래스
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, save_path='best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def load_best(self, model):
        model.load_state_dict(torch.load(self.save_path))
        return model


early_stopping = EarlyStopping(patience=30)


# ============================================================
# 5. 학습 루프
# ============================================================
n_epochs = 500    # 충분히 크게. 어차피 Early Stop이 멈춰줌

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # === 학습 ===
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
    val_loss /= len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # === Early Stopping 체크 ===
    improved = early_stopping(val_loss, model)
    
    if (epoch + 1) % 20 == 0 or improved:
        marker = "✅" if improved else " "
        print(f"{marker} Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break


# 베스트 모델 불러오기
model = early_stopping.load_best(model)


# ============================================================
# 6. Test 평가
# ============================================================
model.eval()
with torch.no_grad():
    pred_t = model(X_test_t.to(device)).cpu().numpy()

# 원래 단위로 되돌리기
pred_original = scaler_y.inverse_transform(pred_t.reshape(-1, 1)).flatten()
target_original = y_test


rmse = np.sqrt(mean_squared_error(target_original, pred_original))
mae = mean_absolute_error(target_original, pred_original)
r2 = r2_score(target_original, pred_original)

print(f"\n[최종 Test 평가]")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")


# ============================================================
# 7. 시각화
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 곡선
best_epoch = val_losses.index(min(val_losses))
axes[0].plot(train_losses, label='Train', color='steelblue')
axes[0].plot(val_losses, label='Val', color='salmon')
axes[0].axvline(best_epoch, color='red', linestyle='--', label=f'Best ({best_epoch+1})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('학습 곡선')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale('log')

# 예측 vs 실제
axes[1].scatter(target_original, pred_original, alpha=0.5, color='steelblue')
axes[1].plot([target_original.min(), target_original.max()],
             [target_original.min(), target_original.max()],
             'r--', lw=2, label='완벽 (y=x)')
axes[1].set_xlabel('실제 집값')
axes[1].set_ylabel('예측 집값')
axes[1].set_title(f'예측 vs 실제\nRMSE={rmse:.2f}, R²={r2:.4f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('full_dl_regression.png', dpi=80)
plt.show()
```

---

## 결과 비교: 무기 적용 전후

같은 데이터, 같은 모델 구조로 무기들을 하나씩 추가하면서 비교한 결과입니다.

| 설정 | Train RMSE | Test RMSE | Gap |
|------|-----------|---------|-----|
| 베이스라인 (무기 0개) | 0.5 | 5.5 | **5.0** (심한 과적합) |
| + Early Stopping | 1.5 | 4.0 | **2.5** |
| + Dropout | 2.0 | 3.5 | **1.5** |
| + Weight Decay | 2.2 | 3.2 | **1.0** (좋음) |

세 무기를 다 쓰면 **Train Loss는 약간 올라가지만, Test Loss가 훨씬 좋아집니다.** 이게 정규화의 본질이에요.

---

## 어떤 무기를 어느 순서로?

회사에서 모델을 만드시면, 다음 순서로 무기를 추가해 보세요.

```
1. 베이스라인 (무기 0개)
   → 학습 곡선 확인
   ↓
2. 과적합 의심? (Train ≪ Test)
   → Early Stopping 추가
   ↓
3. 여전히 과적합?
   → Dropout(0.3) 추가
   ↓
4. 여전히 과적합?
   → Weight Decay (0.01) 추가
   ↓
5. 그래도 과적합?
   → 모델 작게 / 데이터 더 / BatchNorm
```

**한 번에 다 추가하지 마세요.** 하나씩 추가하면서 효과를 봐야 어떤 게 진짜 도움이 됐는지 알 수 있어요.

---

## 다른 정규화 기법들 (참고)

지금까지 배운 세 가지 외에도 더 있어요. 키워드만 알아두세요.

| 기법 | 설명 | 사용 시기 |
|------|------|----------|
| **Data Augmentation** | 데이터를 인위적으로 늘림 | 이미지 (회전, 잘라내기), 텍스트 (어순 변경) |
| **Mixup** | 두 샘플을 섞어서 새 샘플 생성 | 이미지 분류 |
| **Label Smoothing** | 정답을 100%가 아니라 90%로 | 분류 모델 |
| **Gradient Clipping** | 그래디언트가 너무 커지지 못하게 | RNN, Transformer |
| **Layer Normalization** | BatchNorm의 변형 | Transformer |
| **Spectral Normalization** | 가중치 행렬의 특이값 제한 | GAN |

이 자료에서는 안 다룹니다. 더 깊이 가시면 만나게 될 거예요.

---

## 정리

```python
# 표준 패턴 (기억하세요)

# 1. 모델: Dropout + BatchNorm
class Model(nn.Module):
    def __init__(self):
        ...
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(...)

# 2. 옵티마이저: AdamW + weight_decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)

# 3. 학습 루프: Early Stopping
early_stopping = EarlyStopping(patience=20)
for epoch in range(...):
    train()
    val_loss = evaluate()
    if early_stopping(val_loss, model):
        save_best()
    if early_stopping.early_stop:
        break

# 4. 끝나면 베스트 불러오기
model.load_state_dict(torch.load('best.pth'))
```

**이 패턴이 PyTorch 회귀의 90%예요.**

➡️ 다음: [5.7 현업 체크리스트](07-현업-체크리스트.md)
