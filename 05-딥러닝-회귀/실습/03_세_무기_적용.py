"""5장 6절 실습: Early Stopping + Dropout + Weight Decay 종합

세 가지 무기를 다 적용한 표준 패턴.
회사에서 그대로 가져다 쓰셔도 되는 코드입니다.
"""

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
# 시드
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. 데이터: train/val/test 3-way split
# ============================================================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


X_temp, X_test, y_temp, y_test = train_test_split(
    data, target, test_size=0.15, random_state=SEED,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=SEED,
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# 정규화
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


train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)


# ============================================================
# 2. 모델: Dropout + BatchNorm
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


# ============================================================
# 3. Early Stopping 클래스
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_epoch = epoch
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


# ============================================================
# 4. 학습 (AdamW + Weight Decay + Early Stopping)
# ============================================================
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,    # ← Weight Decay
)

early_stopping = EarlyStopping(patience=30)

n_epochs = 500
train_losses = []
val_losses = []

print("\n학습 시작 (Early Stopping이 알아서 멈춤)")
print("-" * 60)

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
            val_loss += loss_fn(model(X), y).item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early Stopping 체크
    improved = early_stopping(val_loss, model, epoch)

    if (epoch + 1) % 20 == 0 or improved:
        marker = "✅" if improved else "  "
        print(f"{marker} Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

    if early_stopping.early_stop:
        print(f"\n→ Early stopping at epoch {epoch+1}")
        print(f"→ Best epoch was {early_stopping.best_epoch+1}")
        break


# 베스트 모델 불러오기
model = early_stopping.load_best(model)


# ============================================================
# 5. Test 평가
# ============================================================
model.eval()
with torch.no_grad():
    pred_t = model(X_test_t.to(device)).cpu().numpy()

pred_original = scaler_y.inverse_transform(pred_t.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test, pred_original))
mae = mean_absolute_error(y_test, pred_original)
r2 = r2_score(y_test, pred_original)

print(f"\n[최종 Test 평가]")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")


# ============================================================
# 6. 시각화
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 학습 곡선 (정상 스케일)
axes[0, 0].plot(train_losses, label='Train', color='steelblue')
axes[0, 0].plot(val_losses, label='Val', color='salmon')
axes[0, 0].axvline(early_stopping.best_epoch, color='red', linestyle='--',
                   label=f'Best ({early_stopping.best_epoch+1})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('학습 곡선')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 학습 곡선 (로그 스케일)
axes[0, 1].plot(train_losses, label='Train', color='steelblue')
axes[0, 1].plot(val_losses, label='Val', color='salmon')
axes[0, 1].axvline(early_stopping.best_epoch, color='red', linestyle='--')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss (log)')
axes[0, 1].set_title('학습 곡선 (로그 스케일)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 예측 vs 실제
axes[1, 0].scatter(y_test, pred_original, alpha=0.5, color='steelblue')
axes[1, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('실제 집값')
axes[1, 0].set_ylabel('예측 집값')
axes[1, 0].set_title(f'예측 vs 실제\nRMSE={rmse:.2f}, R²={r2:.4f}')
axes[1, 0].grid(alpha=0.3)

# 잔차
residuals = y_test - pred_original
axes[1, 1].scatter(pred_original, residuals, alpha=0.5, color='steelblue')
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('예측값')
axes[1, 1].set_ylabel('잔차')
axes[1, 1].set_title('잔차 플롯')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Early Stopping + Dropout + Weight Decay 종합', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('three_weapons.png', dpi=80)
plt.show()
print("\n저장: three_weapons.png")
print("저장: best_model.pth")


# ============================================================
# 7. 정리
# ============================================================
print(f"""
{'=' * 60}
요약:
- 학습 epoch: {len(train_losses)} (최대 {n_epochs}, Early Stop으로 중단)
- 베스트 epoch: {early_stopping.best_epoch+1}
- 베스트 Val Loss: {early_stopping.best_score:.4f}
- Test RMSE: {rmse:.4f}
- Test R²: {r2:.4f}

다음을 적용했습니다:
1. ✅ Early Stopping (patience=30)
2. ✅ Dropout (p=0.3)
3. ✅ Weight Decay (0.01, AdamW)
4. ✅ BatchNorm (덤)

이 패턴이 PyTorch 회귀의 표준이에요.
{'=' * 60}
""")
