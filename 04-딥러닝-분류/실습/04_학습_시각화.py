"""4장 4절 실습: 실시간 학습 시각화

학습이 진행되는 모습을 매 epoch마다 그래프로 보여줍니다.
Colab에서 진짜 멋지게 동작해요.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from IPython.display import clear_output
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


# ============================================================
# 설정
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


# ============================================================
# 데이터
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


# ============================================================
# 모델 (단순)
# ============================================================
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 학습 함수
# ============================================================
def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)

            total_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# 학습 + 매 epoch 시각화
# ============================================================
N_EPOCHS = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []


def update_plot(train_losses, train_accs, test_losses, test_accs, current_epoch):
    """매 epoch마다 그래프 갱신."""
    if IN_NOTEBOOK:
        clear_output(wait=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(train_losses, 'o-', label='Train', color='steelblue')
    axes[0].plot(test_losses, 's-', label='Test', color='salmon')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss (Epoch {current_epoch})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, 'o-', label='Train', color='steelblue')
    axes[1].plot(test_accs, 's-', label='Test', color='salmon')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy (Epoch {current_epoch})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 진단 텍스트
    if len(test_accs) > 0:
        gap = train_accs[-1] - test_accs[-1]
        if gap > 0.05:
            status = f"⚠️ 과적합 의심 (gap={gap:.3f})"
        elif gap < 0.01:
            status = f"✅ 정상 (gap={gap:.3f})"
        else:
            status = f"🟡 약간 과적합 (gap={gap:.3f})"
        plt.suptitle(status, fontsize=14)

    plt.tight_layout()
    plt.show()


for epoch in range(N_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{N_EPOCHS}  "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}  "
          f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    update_plot(train_losses, train_accs, test_losses, test_accs, epoch+1)


# ============================================================
# 최종 그래프 저장
# ============================================================
plt.savefig('final_curves.png', dpi=80)
print(f"\n최종 정확도: {test_accs[-1]:.4f}")
print("저장: final_curves.png")
