# 4.4 학습 과정 들여다보기

> **선행 학습:** 신경망의 학습은 **미분**을 기반으로 합니다.  
> [👉 미분 입문](../../부록/수학/03-미분-입문.md) — 기울기와 최적화  
> [👉 선형대수 입문](../../부록/수학/04-선형대수-입문.md) — 행렬 계산  
> 단, 코드부터 배우고 나중에 수학을 깊이 이해해도 괜찮습니다.

---

## "학습이 잘 되고 있는지 어떻게 알아요?"

학습 코드를 돌리면 loss와 accuracy가 화면에 줄줄 출력돼요. 하지만 그 숫자만 봐서는 무슨 일이 일어나는지 감이 안 와요.

이 글에서는 학습 과정을 **시각적으로** 들여다보는 법을 배웁니다.

---

## 학습 곡선 그리기

가장 기본적이고 가장 강력한 도구예요. **매 epoch의 loss와 accuracy를 그래프로 그리는 것.**

### 학습 코드에 기록 추가

```python
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(n_epochs):
    # === 학습 ===
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
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
    
    print(f"Epoch {epoch+1:2d}  "
          f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.4f}  | "
          f"Test Loss: {test_losses[-1]:.4f}  Acc: {test_accs[-1]:.4f}")
```

### 그래프로 그리기

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss 곡선
axes[0].plot(train_losses, label='Train', marker='o')
axes[0].plot(test_losses, label='Test', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('학습 vs 평가 Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy 곡선
axes[1].plot(train_accs, label='Train', marker='o')
axes[1].plot(test_accs, label='Test', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('학습 vs 평가 Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

이 그래프 두 개로 모델의 상태를 진단할 수 있어요.

---

## 곡선 패턴 읽는 법

### 패턴 1: 정상

```
Loss                      Accuracy
 │\                         │      ___
 │ \___                     │  ___/
 │     \___                 │ /
 │         \___             │/
 └─────────────             └─────────────
   Train과 Test가              둘 다 높고
   같이 떨어짐                  비슷한 수준
```

이게 이상적이에요. 학습이 잘 됐고 일반화도 잘 됨.

### 패턴 2: 과적합 (Overfitting)

```
Loss                      Accuracy
 │\                         │      ___
 │ \                        │ ____/
 │  \___ Train              │/  ↗ Test ↘
 │      \___                │
 │ ─────── Test (안 떨어짐)  │
 └─────────────             └─────────────
   Train Loss는 떨어지는데      Train은 100%
   Test Loss는 안 떨어짐        Test는 정체나 하락
```

**과적합이에요.** 모델이 학습 데이터를 외우기 시작했어요. 해결책:
- 학습 일찍 멈추기 (Early Stopping, 5장)
- Dropout 추가
- 모델 단순화
- 데이터 더 모으기

### 패턴 3: 과소적합 (Underfitting)

```
Loss                      Accuracy
 │___                       │
 │   ───                    │  ↗ ↗ ↗
 │      ───                 │ /     (낮은 수준에서 멈춤)
 │ ─────                    │
 └─────────────             └─────────────
   Train Loss도              Train, Test 둘 다
   떨어지다 멈춤               낮은 수준
```

**모델이 너무 단순하거나 학습이 부족한 상태**예요. 해결책:
- 모델 더 크게/깊게
- 학습 더 길게
- 학습률 조정
- 더 좋은 옵티마이저

### 패턴 4: 학습 안 됨 (Loss 폭발 또는 NaN)

```
Loss
 │       /
 │      /
 │     /
 │    /
 │   /
 │  /
 │ /
 └─────────────
   Loss가 점점 커짐 (또는 NaN)
```

**뭔가 크게 잘못됐어요.** 흔한 원인:
- 학습률이 너무 큼 (lr=10 같이)
- 데이터에 NaN 또는 inf 있음
- 정규화 안 함

해결: 학습률을 1/10로 줄여보고, 데이터 확인.

---

## 두 곡선의 차이가 알려주는 것

**Train과 Test의 차이(gap)를 항상 보세요.**

| Gap | 진단 |
|-----|------|
| 작음 (~1%) | 좋음, 학습 더 해도 됨 |
| 보통 (5%) | 약간 과적합, 정상적 |
| 큼 (>10%) | 명백한 과적합 |
| 음수 (Test가 더 잘) | 운 좋음 (또는 우연), 그냥 정상 |

---

## 실시간 시각화 (선택)

학습 중에 매 epoch 그래프를 갱신해 가면서 보고 싶으시면:

```python
from IPython.display import clear_output

for epoch in range(n_epochs):
    # ... 학습 코드 ...
    
    # 매 epoch마다 그래프 갱신
    clear_output(wait=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(test_losses, label='Test')
    axes[0].legend()
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(test_accs, label='Test')
    axes[1].legend()
    plt.show()
```

Colab에서 진짜 멋지게 동작해요. 학습이 어떻게 진행되는지 한눈에 보입니다.

---

## TensorBoard (현업 표준)

Matplotlib으로도 충분하지만, 본격적으로 실험하실 땐 **TensorBoard**가 좋아요.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/my_experiment')

for epoch in range(n_epochs):
    # ... 학습 코드 ...
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)

writer.close()
```

Colab에서 보려면:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

웹 인터페이스에서 그래프, 모델 구조, 가중치 분포 등 다 볼 수 있어요.

---

## Confusion Matrix로 클래스별 성능 보기

전체 정확도가 97%여도, 어느 한 클래스에서 계속 망치고 있을 수 있어요. 혼동 행렬로 확인해 봅시다.

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 모든 예측 모으기
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

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('MNIST 혼동 행렬 (신경망)')
plt.show()
```

대각선이 진하고 다른 칸이 엷으면 좋은 거예요. 특정 칸이 진하면 그 부분에서 자주 헷갈리는 거예요.

---

## 잘못 분류된 샘플 보기

```python
import numpy as np

wrong_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
print(f"틀린 개수: {len(wrong_indices)}")

# 처음 10개 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    if i >= 10:
        break
    idx = wrong_indices[i]
    img, _ = test_dataset[idx]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'정답:{all_labels[idx]} 예측:{all_preds[idx]}', fontsize=10)
    ax.axis('off')

plt.suptitle('잘못 분류된 샘플들', fontsize=14)
plt.tight_layout()
plt.show()
```

이걸 보면 **모델이 사람이 봐도 헷갈릴 만한 글씨에서 틀린다**는 걸 확인할 수 있어요. 정상이에요. 사람도 100%는 못 맞춥니다.

---

## 가중치 분포 보기 (가끔 유용)

학습 후 가중치들이 어떻게 분포되어 있는지 보면 모델의 건강 상태를 알 수 있어요.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    weights = layer.weight.data.cpu().numpy().flatten()
    axes[i].hist(weights, bins=50, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Layer {i+1} 가중치 분포')
    axes[i].set_xlabel('가중치 값')

plt.tight_layout()
plt.show()
```

좋은 신호:
- 0 주변에 집중 (정규분포 같은 모양)
- 너무 크지도 작지도 않음

나쁜 신호:
- 가중치가 극단적으로 큼 → gradient explosion 의심
- 모두 0 근처 → gradient vanishing 의심

---

## 학습률 (learning rate) 효과

학습률은 가장 중요한 하이퍼파라미터예요. 같은 모델/데이터로 lr만 바꿔보면:

```
lr = 0.0001 (너무 작음): 학습이 너무 느림
lr = 0.001  (보통): 잘 됨
lr = 0.01   (좀 큼): 빨리 배우지만 불안정
lr = 0.1    (너무 큼): Loss가 발산하거나 nan
lr = 1.0    (망함): 처음부터 nan
```

**Adam의 기본값 0.001부터 시작**하시고, 학습이 잘 안 되면 ÷ 10 (0.0001) 또는 × 10 (0.01)을 시도해 보세요.

### Learning Rate Scheduler

학습이 진행되면서 lr을 점차 줄이는 게 효과적이에요.

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# 5 epoch마다 lr을 절반으로

for epoch in range(n_epochs):
    # ... 학습 ...
    scheduler.step()
```

또는 검증 점수가 안 좋아질 때만 줄이는 `ReduceLROnPlateau`도 있어요.

---

## 정리: 학습 진단 체크리스트

학습 후에 항상 다음을 확인하세요.

```
[ ] Loss 곡선 그렸는가
    → Train, Test 둘 다 떨어지는가
    → Test가 Train보다 너무 높지 않은가 (과적합)

[ ] Accuracy 곡선 그렸는가
    → 학습이 정체된 시점이 있는가

[ ] 혼동 행렬 봤는가
    → 특정 클래스에서 자주 망치고 있지 않은가

[ ] 잘못 분류된 샘플 몇 개 봤는가
    → 사람도 헷갈릴 만한 것인가, 아니면 명백히 어긋나는가

[ ] 학습률 적절한가
    → Loss가 빠르게 떨어지다가 정체 → lr ↓
    → Loss가 폭발 → lr ↓↓
    → Loss가 너무 천천히 떨어짐 → lr ↑
```

---

## 자주 묻는 질문

> **Q. Epoch을 얼마나 해야 하나요?**
>
> Test loss가 계속 떨어지면 → 더 학습.
> Test loss가 정체되거나 올라가면 → 멈춤.
> 자동화하려면 **Early Stopping** (5장에서)을 쓰세요.

> **Q. 한 epoch에 몇 분이 걸리나요?**
>
> MNIST는 GPU에서 1 epoch에 ~10초. 큰 데이터셋(ImageNet)은 1 epoch에 몇 시간이 걸려요. 자기 데이터에 맞춰 시간 예산을 짜세요.

> **Q. Loss가 nan이 됐어요!**
>
> 거의 항상 학습률 때문이에요. lr을 1/10로 줄여보세요. 그래도 안 되면 데이터에 nan 있는지 확인.

➡️ 다음: [4.5 더 깊게: 활성화/손실/옵티마이저](05-깊게.md)
