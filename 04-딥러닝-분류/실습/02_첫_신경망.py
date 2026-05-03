"""4장 3~5절 실습: MNIST를 신경망으로 분류

PyTorch의 표준 학습 루프를 익히세요.
이 패턴이 모든 딥러닝 코드의 기본입니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 시드 고정 (재현성)
# ============================================================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# ============================================================
# 설정
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 0.001


# ============================================================
# 1. 데이터 준비
# ============================================================
print("\n[1] 데이터 준비")

transform = transforms.Compose([
    transforms.ToTensor(),    # 0~255 → 0~1
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform,
)
test_dataset = datasets.MNIST(
    './data', train=False, download=True, transform=transform,
)

print(f"학습: {len(train_dataset)}개, 평가: {len(test_dataset)}개")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
)


# ============================================================
# 2. 모델 정의
# ============================================================
print("\n[2] 모델 정의")


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


model = MNISTClassifier().to(device)
print(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터: {n_params:,}")


# ============================================================
# 3. 손실 + 옵티마이저
# ============================================================
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)


# ============================================================
# 4. 학습 루프
# ============================================================
print("\n[3] 학습 시작")

train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_test_acc = 0.0

for epoch in range(N_EPOCHS):
    # === 학습 ===
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 표준 5단계
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    # === 평가 ===
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    # 기록
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_correct / train_total)
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(test_correct / test_total)

    print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  "
          f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.4f}  | "
          f"Test Loss: {test_losses[-1]:.4f}  Acc: {test_accs[-1]:.4f}")

    # 베스트 모델 저장
    if test_accs[-1] > best_test_acc:
        best_test_acc = test_accs[-1]
        torch.save(model.state_dict(), 'best_mnist_model.pth')


print(f"\n베스트 정확도: {best_test_acc:.4f}")
print(f"마지막 정확도: {test_accs[-1]:.4f}")


# ============================================================
# 5. 학습 곡선 시각화
# ============================================================
print("\n[4] 학습 곡선")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(train_losses, label='Train', marker='o')
axes[0].plot(test_losses, label='Test', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss 곡선')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(train_accs, label='Train', marker='o')
axes[1].plot(test_accs, label='Test', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy 곡선')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=80)
plt.show()
print("저장: training_curves.png")


# ============================================================
# 6. 베스트 모델로 자세한 평가
# ============================================================
print("\n[5] 자세한 평가")

# 베스트 모델 불러오기
model.load_state_dict(torch.load('best_mnist_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('MNIST 혼동 행렬 (신경망)')
plt.tight_layout()
plt.savefig('cm_dl.png', dpi=80)
plt.show()
print("저장: cm_dl.png")

print("\n[분류 리포트]")
print(classification_report(all_labels, all_preds))


# ============================================================
# 7. 잘못 분류된 샘플
# ============================================================
wrong_idx = np.where(all_preds != all_labels)[0]
print(f"\n틀린 개수: {len(wrong_idx)}")

n_show = min(10, len(wrong_idx))
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    if i >= n_show:
        ax.axis('off')
        continue
    idx = wrong_idx[i]
    img, _ = test_dataset[idx]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'정답:{all_labels[idx]} 예측:{all_preds[idx]}', fontsize=10)
    ax.axis('off')

plt.suptitle('잘못 분류된 샘플들', fontsize=14)
plt.tight_layout()
plt.savefig('wrong_dl.png', dpi=80)
plt.show()


print("\n첫 신경망 끝! SVM의 91%를 신경망이 97%+로 끌어올렸어요.")
print(f"\n베스트 모델 저장됨: best_mnist_model.pth")
