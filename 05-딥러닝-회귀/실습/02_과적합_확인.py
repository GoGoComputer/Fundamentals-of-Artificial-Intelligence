"""5장 2절 실습: 과적합을 일부러 만들어 보기

작은 데이터에 큰 모델을 너무 많이 학습시키면 과적합이 일어납니다.
직접 만들어 보면서 과적합의 모습을 눈으로 확인하세요.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 진짜 패턴: y = sin(x) + noise
# ============================================================
n_samples = 30   # 일부러 적게!
X = np.linspace(0, 2*np.pi, n_samples)
y = np.sin(X) + np.random.normal(0, 0.1, n_samples)

X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y, dtype=torch.float32)


# ============================================================
# 너무 큰 모델 (과적합 유도)
# ============================================================
class TooBigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x).squeeze()


# ============================================================
# 적당한 모델
# ============================================================
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x).squeeze()


# ============================================================
# 학습 함수
# ============================================================
def train_model(model, X, y, n_epochs=5000, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# 학습
print("[큰 모델 학습 중...]")
big_model = TooBigModel()
big_losses = train_model(big_model, X_t, y_t)
print(f"  최종 Train Loss: {big_losses[-1]:.6f}")

print("\n[적당한 모델 학습 중...]")
good_model = GoodModel()
good_losses = train_model(good_model, X_t, y_t)
print(f"  최종 Train Loss: {good_losses[-1]:.6f}")


# ============================================================
# 시각화
# ============================================================
X_plot = np.linspace(0, 2*np.pi, 200)
X_plot_t = torch.tensor(X_plot, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    big_pred = big_model(X_plot_t).numpy()
    good_pred = good_model(X_plot_t).numpy()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 과적합 모델
axes[0].scatter(X, y, color='red', s=80, label='학습 데이터', zorder=5, edgecolor='black')
axes[0].plot(X_plot, np.sin(X_plot), 'g--', label='진짜 함수 (sin)', linewidth=2)
axes[0].plot(X_plot, big_pred, 'b-', label='큰 모델 예측', linewidth=2)
axes[0].set_title(f'과적합 모델\n(파라미터 ~130K, 학습점 30개)\nLoss={big_losses[-1]:.6f}')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# 적당한 모델
axes[1].scatter(X, y, color='red', s=80, label='학습 데이터', zorder=5, edgecolor='black')
axes[1].plot(X_plot, np.sin(X_plot), 'g--', label='진짜 함수 (sin)', linewidth=2)
axes[1].plot(X_plot, good_pred, 'b-', label='적당한 모델 예측', linewidth=2)
axes[1].set_title(f'적당한 모델\n(파라미터 ~50, 학습점 30개)\nLoss={good_losses[-1]:.6f}')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('overfitting_demo.png', dpi=80)
plt.show()
print("\n저장: overfitting_demo.png")


# ============================================================
# 학습 곡선
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(big_losses, label='큰 모델 (과적합)', color='red', alpha=0.7)
plt.plot(good_losses, label='적당한 모델', color='blue', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('학습 곡선 비교 (Train Loss만)')
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('overfitting_curves.png', dpi=80)
plt.show()


# ============================================================
# 결론
# ============================================================
print("""
관찰할 점:
1. 큰 모델은 모든 학습 점을 정확히 맞히지만 (Loss 매우 작음),
   학습 점 사이에서 미친 듯이 흔들립니다.

2. 적당한 모델은 학습 점을 약간 틀리지만 (Loss 약간 큼),
   진짜 함수(sin)에 가까운 부드러운 곡선을 만들어요.

3. 새로운 점에서 어느 모델이 더 정확할까요? 적당한 모델이에요.
   이게 일반화(generalization)의 의미입니다.

4. 큰 모델로 같은 효과를 내려면 다음 중 하나 또는 여러 개:
   - Early Stopping
   - Dropout
   - Weight Decay
   - 데이터 더 모으기
""")
