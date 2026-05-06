# 4.7 현업 체크리스트

> ML 체크리스트([2.8](../../02-머신러닝-분류/이론/08-현업-체크리스트.md))는 딥러닝에도 거의 다 적용돼요.
> 이 글은 **딥러닝 특유의 체크 항목**만 다룹니다.

---

## 1. 재현성 (Reproducibility)

딥러닝은 ML보다 훨씬 더 **랜덤성**에 의존해요. 가중치 초기화, 데이터 셔플, 드롭아웃 등. 그래서 같은 코드를 두 번 돌려도 결과가 달라집니다.

### 시드 모두 고정

```python
import torch
import numpy as np
import random

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# CuDNN 결정론적 모드
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

⚠️ `cudnn.deterministic=True`는 **속도가 약간 느려져요.** 실험 단계엔 켜두고, 최종 학습엔 끌 수도 있어요.

### DataLoader도 시드 고정

```python
g = torch.Generator()
g.manual_seed(SEED)

loader = DataLoader(dataset, ..., generator=g)
```

---

## 2. Validation Set 만들기

ML에서는 train/test 두 개로 충분했지만, 딥러닝은 **세 개**로 나누는 게 보통이에요.

```python
from torch.utils.data import random_split

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

# DataLoader 따로
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

- **train**: 학습에만 사용
- **val**: 매 epoch마다 평가, 하이퍼파라미터 튜닝에 사용
- **test**: 최종 평가만, 다른 곳엔 절대 안 씀

---

## 3. Best Model 저장

학습 중 **검증 성능이 가장 좋았던 시점의 모델**을 저장하세요.

```python
best_val_acc = 0.0

for epoch in range(n_epochs):
    # 학습
    train_one_epoch(model, train_loader)
    
    # 검증
    val_acc = evaluate(model, val_loader)
    
    # 베스트면 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch}: 베스트 갱신 (val_acc={val_acc:.4f})")

# 학습 끝나면 베스트 불러오기
model.load_state_dict(torch.load('best_model.pth'))
```

이렇게 하면 **마지막 epoch이 아니라 가장 좋았던 시점의 모델**을 얻을 수 있어요.

---

## 4. Early Stopping

검증 성능이 일정 epoch 동안 안 좋아지면 자동으로 멈춤.

```python
patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(n_epochs):
    train_one_epoch(...)
    val_loss = evaluate_loss(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

자동화 도구도 있어요 (PyTorch Lightning 등).

[5장](../../05-딥러닝-회귀/)에서 자세히 다룹니다.

---

## 5. 학습 곡선 모니터링

매 학습마다 그래프를 그려서 저장하세요.

```python
import matplotlib.pyplot as plt

def plot_history(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
```

매 실험의 학습 곡선을 보존하면 **나중에 비교**할 수 있어요.

---

## 6. 실험 추적 (MLflow, W&B)

딥러닝은 실험을 수십 번 하게 됩니다. 손으로 결과를 기록하면 금방 망해요.

### Weights & Biases (W&B) — 가장 인기

```python
import wandb

wandb.init(
    project="mnist-classification",
    config={
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "model": "MLP",
    },
)

for epoch in range(n_epochs):
    train_loss = train_one_epoch(...)
    val_loss = evaluate(...)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

wandb.finish()
```

웹 대시보드에서 그래프, 비교, 정렬 다 됩니다. **무료 버전으로 충분**해요.

---

## 7. Checkpoint 저장 (긴 학습용)

학습이 며칠 걸리는 경우, 중간에 컴퓨터가 죽으면 처음부터 다시 해야 하면 끔찍해요. Checkpoint를 저장하세요.

```python
# 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_acc': best_val_acc,
}
torch.save(checkpoint, f'checkpoint_epoch{epoch}.pth')

# 불러와서 이어서 학습
checkpoint = torch.load('checkpoint_epoch10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, n_epochs):
    # 이어서 학습
    ...
```

---

## 8. 학습률 스케줄링

학습률을 시간에 따라 조정하면 보통 더 좋은 결과가 나와요.

### Cosine Annealing (요즘 표준)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

for epoch in range(n_epochs):
    train_one_epoch(...)
    scheduler.step()
    
    print(f"Current LR: {scheduler.get_last_lr()[0]}")
```

학습률이 코사인 곡선처럼 천천히 줄어들어요.

### ReduceLROnPlateau (검증 점수 기반)

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

for epoch in range(n_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)
    scheduler.step(val_loss)    # val_loss가 안 좋아지면 lr 줄임
```

---

## 9. 모델 크기와 메모리

모델 파라미터 수와 메모리를 알아두세요.

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"파라미터 수: {count_parameters(model):,}")

# 또는 torchinfo
from torchinfo import summary
summary(model, input_size=(1, 1, 28, 28))
```

| 모델 크기 | 예시 |
|----------|------|
| 작음 (< 1M) | 우리의 MLP |
| 중간 (1~100M) | ResNet-50 (25M) |
| 큼 (100M~1B) | BERT-Base (110M) |
| 거대 (1B+) | GPT-3 (175B), GPT-4 (??) |

큰 모델은:
- 더 큰 GPU 필요 (VRAM)
- 더 많은 학습 데이터 필요
- 더 긴 학습 시간

---

## 10. 모델 디스크 형식

PyTorch 모델 저장 형식 비교:

| 형식 | 사용처 |
|------|------|
| `.pth` (state_dict) | PyTorch 기본 |
| `.onnx` | 다른 프레임워크/하드웨어 호환 |
| `TorchScript` (`.pt`) | C++/모바일 배포 |
| `.gguf` | LLM 양자화 (llama.cpp 등) |

학습 단계에서는 `.pth`로 충분해요. 배포 시에는 ONNX 변환이 자주 쓰여요.

---

## 11. Mixed Precision (FP16)

[GPU 활용](06-GPU-활용.md)에서 봤죠. 큰 모델 학습할 땐 거의 표준이에요.

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with autocast():
        output = model(x)
        loss = loss_fn(output, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 12. 추론 시 최적화

배포 단계에서는 학습과 다른 최적화가 필요해요.

### Eval 모드와 no_grad

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

이 두 가지를 안 하면 **메모리 5배, 속도 2배 느려져요.**

### Half precision 추론

```python
model = model.half()    # FP16으로
x = x.half()
with torch.no_grad():
    output = model(x)
```

### TorchScript / ONNX

학습은 PyTorch로, 배포는 변환된 형식으로 하면 빨라져요.

```python
# TorchScript
model.eval()
example = torch.rand(1, 1, 28, 28)
traced = torch.jit.trace(model, example)
traced.save('model.pt')

# ONNX
torch.onnx.export(model, example, 'model.onnx')
```

---

## 딥러닝 현업 체크리스트

```
[ ] 시드 모두 고정 (재현성)
[ ] train/val/test 셋 분리
[ ] best model 저장
[ ] Early Stopping 구현
[ ] 학습 곡선 시각화 + 저장
[ ] 실험 추적 (W&B 또는 MLflow)
[ ] Checkpoint 저장 (긴 학습)
[ ] 학습률 스케줄링
[ ] 모델 파라미터 수 확인
[ ] Mixed Precision 적용 (큰 모델)
[ ] 추론 시 model.eval() + no_grad()
[ ] 배포 형식 결정 (.pth, ONNX, TorchScript)
[ ] GPU 메모리 모니터링
```

---

## 4장을 마치며

딥러닝 분류, 끝났습니다!

이 장에서 배운 것:
- ✅ 신경망의 직관 (곱하고 더하고, 활성화)
- ✅ PyTorch 기본 (텐서, GPU, autograd, nn.Module)
- ✅ 학습 루프의 6단계 (zero_grad → forward → loss → backward → step)
- ✅ 학습 과정 시각화 + 진단
- ✅ 활성화/손실/옵티마이저의 선택
- ✅ GPU 활용
- ✅ 현업 코드 작성

다음 5장은 **딥러닝 회귀**예요. 이 장에서 배운 거의 모든 게 적용되고, **과적합과 진짜로 싸우는 법**을 배우게 됩니다.

➡️ **[5장. 딥러닝 — 회귀](../../05-딥러닝-회귀/)**
