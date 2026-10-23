# 7.7 강화학습 — 시행착오로 배우는 AI

## 들어가며

지금까지 다룬 머신러닝은 모두 **정답이 있는 데이터**에서 출발했습니다.
- 지도학습: 입력과 정답 라벨
- 비지도학습: 라벨은 없지만 데이터 자체가 주어짐

강화학습(Reinforcement Learning, RL)은 다릅니다. **정답을 모릅니다.** 대신 환경과 상호작용하면서 "보상"이라는 신호만 받습니다. 무엇을 해야 보상이 클지는 직접 시도해보며 알아내야 합니다.

마치 어린아이가 자전거를 배우는 과정과 비슷합니다. 누가 페달 각도와 균형 잡는 법을 수학으로 알려주지 않습니다. 넘어지고(음의 보상), 몇 미터 굴러가고(양의 보상), 점점 균형을 익힙니다.

이 패러다임으로 만들어진 결과물들이 인상적입니다.
- 알파고가 이세돌을 이긴 방식
- ChatGPT가 사람 선호에 맞춰 답변하는 방식 (RLHF)
- 로봇이 걷고 물건을 집는 방식
- 자율주행차가 차선 변경을 결정하는 방식

---

## 1. 강화학습의 기본 구성요소

### 1.1 다섯 가지 주인공

| 용어 | 의미 | 게임 비유 |
|------|------|-----------|
| Agent (에이전트) | 학습하는 주체 | 플레이어 |
| Environment (환경) | 에이전트가 작동하는 세계 | 게임 |
| State (상태 `s`) | 현재 상황의 정보 | 화면 |
| Action (행동 `a`) | 에이전트가 하는 선택 | 버튼 입력 |
| Reward (보상 `r`) | 행동의 결과로 받는 신호 | 점수 |

기본 흐름은 이렇게 반복됩니다.

```
상태 s_t 관찰 → 행동 a_t 선택 → 보상 r_t 받음 → 새 상태 s_{t+1}
   ↑                                                      |
   └──────────────────────────────────────────────────────┘
```

### 1.2 정책(Policy)

에이전트의 의사결정 방식을 **정책(π)**이라고 합니다. "이 상태에서는 이 행동을 한다"는 함수입니다.

- 결정적 정책: π(s) = a (상태마다 행동 하나)
- 확률적 정책: π(a|s) = 확률 (행동의 확률 분포)

학습의 목표는 결국 "기대 보상을 최대화하는 정책 π*를 찾기"입니다.

### 1.3 누적 보상과 할인율

순간의 보상보다 **장기 누적 보상**이 중요합니다. 미래 보상은 보통 할인율 γ(0~1)를 곱해 현재 가치로 환산합니다.

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots$$

γ가 0이면 근시안적(지금 보상만 신경 씀), 1에 가까우면 장기 시야. 보통 0.9~0.99를 씁니다.

---

## 2. 마르코프 결정 과정 (MDP)

강화학습이 다루는 환경의 표준 수학적 표현은 **Markov Decision Process(MDP)**입니다.

### 2.1 마르코프 성질

> 미래는 오직 현재 상태에만 의존한다. 과거 이력은 불필요하다.

체스를 생각해보세요. 현재 보드 상태만 보면 다음 수를 둘 수 있습니다. 이전에 어떤 순서로 말이 움직였는지 알 필요가 없습니다.

이 가정이 깨지는 환경(예: 일부만 보이는 게임)은 POMDP(Partially Observable MDP)라고 따로 다룹니다.

### 2.2 가치 함수

두 종류의 가치 함수가 있습니다.

- **상태 가치 함수 V(s)**: "이 상태에서 시작해 정책 π를 따랐을 때 기대되는 누적 보상"
- **행동 가치 함수 Q(s, a)**: "이 상태에서 이 행동을 한 뒤 정책 π를 따랐을 때 기대되는 누적 보상"

Q가 더 실용적입니다. Q(s, a)를 알면 "각 행동의 가치를 비교해서 최댓값을 고르면" 됩니다.

### 2.3 벨만 방정식

가치 함수는 재귀적으로 표현됩니다.

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

"지금 받는 보상 + 다음 상태에서 가장 좋은 행동의 가치"라는 뜻입니다. 이 식이 강화학습 모든 알고리즘의 출발점입니다.

---

## 3. Q-Learning — 가장 단순한 알고리즘

### 3.1 핵심 아이디어

상태와 행동의 모든 조합에 대해 Q값을 표(Q-table)로 만들어 학습합니다.

업데이트 규칙:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]$$

α는 학습률입니다. 새 추정치(`r + γ max Q(s',a')`)와 기존 추정치(`Q(s,a)`)의 차이를 일부 반영합니다.

### 3.2 탐험과 활용의 딜레마

- **활용 (Exploitation)**: 지금까지 배운 최선의 행동 선택
- **탐험 (Exploration)**: 새로운 행동 시도

탐험을 안 하면 더 좋은 전략을 못 찾습니다. 탐험만 하면 학습이 안 됩니다. 균형이 필요합니다.

가장 흔한 방법은 **ε-greedy**입니다.

```python
import random

def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)        # 탐험: 무작위
    else:
        return max(actions, key=lambda a: Q[state, a])  # 활용: 최선
```

학습 초반엔 ε을 높게(예: 1.0) 시작해 탐험을 많이 하고, 점차 줄여서(예: 0.05) 학습된 전략을 활용합니다.

### 3.3 그리드월드로 직접 구현

`5×5` 격자에서 시작점에서 목표점까지 가는 에이전트를 학습시켜봅니다.

```python
import numpy as np
import random

# 환경: 5x5 격자, 목표(4,4)에 도달하면 +10, 매 스텝 -1
GRID_SIZE = 5
GOAL = (4, 4)
START = (0, 0)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 상하좌우

def step(state, action):
    new_r = max(0, min(GRID_SIZE-1, state[0] + action[0]))
    new_c = max(0, min(GRID_SIZE-1, state[1] + action[1]))
    new_state = (new_r, new_c)
    if new_state == GOAL:
        return new_state, 10, True
    return new_state, -1, False

# Q-table 초기화
Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# 하이퍼파라미터
alpha = 0.1
gamma = 0.95
epsilon = 1.0
episodes = 500

for ep in range(episodes):
    state = START
    done = False
    while not done:
        # ε-greedy 행동 선택
        if random.random() < epsilon:
            a_idx = random.randint(0, 3)
        else:
            a_idx = np.argmax(Q[state[0], state[1]])

        new_state, reward, done = step(state, ACTIONS[a_idx])

        # Q 업데이트
        best_next = np.max(Q[new_state[0], new_state[1]])
        td_target = reward + gamma * best_next * (not done)
        Q[state[0], state[1], a_idx] += alpha * (td_target - Q[state[0], state[1], a_idx])

        state = new_state

    # ε 감소
    epsilon = max(0.05, epsilon * 0.99)

# 학습된 정책 출력
arrows = ["↑", "↓", "←", "→"]
for r in range(GRID_SIZE):
    line = ""
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            line += " G "
        else:
            line += f" {arrows[np.argmax(Q[r, c])]} "
    print(line)
```

500 에피소드 후 출력은 모든 칸에서 목표 방향을 가리키는 화살표가 됩니다. 어떤 표 형태의 환경에서든 같은 코드 패턴이 작동합니다.

### 3.4 한계 — 차원의 저주

Q-table은 상태와 행동이 적을 때만 가능합니다. 아타리 게임 화면(`84×84` 픽셀, 4프레임 스택)이면 상태 수가 `256^(84×84×4) ≈ 천문학적` 입니다. 표로 못 만듭니다.

이 문제를 풀려고 등장한 게 다음 절의 DQN입니다.

---

## 4. DQN — Q-함수를 신경망으로

### 4.1 발상

> Q-table 대신, 상태를 입력으로 받아 각 행동의 Q값을 출력하는 신경망을 쓰자.

```
입력: 상태 (이미지 등)
   ↓
[Conv → Conv → FC → FC]  ← 신경망
   ↓
출력: Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)
```

### 4.2 학습 안정화 트릭들

신경망을 그냥 쓰면 학습이 매우 불안정합니다. DQN은 두 가지 트릭으로 이를 해결했습니다.

**1) Replay Buffer (경험 재생)**

매 스텝의 `(s, a, r, s', done)`을 메모리에 저장하고, 학습할 땐 무작위로 미니배치 샘플링합니다.

연속한 경험은 강한 상관이 있어 학습을 망칩니다. 무작위 샘플링이 이 상관을 끊어줍니다.

**2) Target Network**

업데이트 타깃을 계산할 때 **느리게 업데이트되는 별도 네트워크**를 씁니다.

```python
# 매 스텝
loss = (Q(s, a) - (r + γ * max Q_target(s', a')))^2
        ↑                       ↑
    main network          target network (천천히 동기화)
```

타깃이 계속 흔들리면 학습이 발산하는데, target network를 천천히 동기화하면 안정됩니다.

### 4.3 PyTorch로 간단히

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=10000)
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.q_net(torch.tensor(state, dtype=torch.float32))
            return q.argmax().item()

    def remember(self, *transition):
        self.buffer.append(transition)

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        s_next = torch.tensor(np.array(s_next), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s_next).max(1)[0]
        q_target = r + self.gamma * q_next * (1 - done)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
```

`gymnasium` 라이브러리의 CartPole 환경에서 100~300 에피소드면 막대를 잘 세우는 정책을 학습합니다.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2)

for episode in range(300):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    if episode % 10 == 0:
        agent.update_target()
    agent.epsilon = max(0.05, agent.epsilon * 0.995)
    print(f"Episode {episode}: reward={total_reward}, eps={agent.epsilon:.3f}")
```

---

## 5. 정책 경사법 (Policy Gradient)

DQN은 가치 함수를 학습한 후 간접적으로 정책을 만듭니다. 정책 경사법은 **정책을 직접 신경망으로 표현**하고 그래디언트로 학습합니다.

### 5.1 REINFORCE 알고리즘

가장 단순한 정책 경사법입니다.

수학적으로 목표는 기대 보상을 최대화하는 것이고, 그래디언트는 다음과 같이 표현됩니다.

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t\big]$$

직관적 의미: "보상이 컸던 행동의 확률을 올리고, 보상이 작았던 행동의 확률을 내린다."

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

policy = PolicyNetwork(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for episode in range(500):
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        probs = policy(torch.tensor(state, dtype=torch.float32))
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        log_probs.append(m.log_prob(action))
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
        state = next_state

    # 누적 보상 계산
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 표준화

    # 손실 = -Σ(log π(a|s) * G)
    loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns)]).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.2 Actor-Critic

REINFORCE는 분산이 커서 학습이 불안정합니다. **Actor-Critic**은 두 네트워크를 같이 씁니다.

- **Actor (정책)**: 행동 결정
- **Critic (가치)**: 행동의 가치 평가 → Actor 학습의 기준

이 구조를 발전시킨 것이 PPO, A3C, SAC, TD3 같은 현대 알고리즘들입니다. PPO는 OpenAI가 ChatGPT의 RLHF에 쓴 바로 그 알고리즘입니다.

---

## 6. PPO — 현대 강화학습의 표준

PPO(Proximal Policy Optimization, 2017)는 학습 안정성과 단순함을 동시에 잡은 알고리즘으로, 현재 가장 널리 쓰이는 RL 방법입니다.

### 6.1 핵심 아이디어

정책을 한 번에 너무 크게 바꾸면 망가집니다. PPO는 "이전 정책과 너무 멀어지지 않게" 클리핑으로 제한합니다.

$$L^{CLIP}(\theta) = \mathbb{E}\Big[\min\big(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big]$$

- `r_t(θ)`: 새 정책과 옛 정책의 비율
- `A_t`: 어드밴티지(이 행동이 평균보다 얼마나 좋았는가)
- `clip`: 비율을 `[1-ε, 1+ε]`로 제한

### 6.2 Stable Baselines3로 PPO 쓰기

직접 구현보다 검증된 라이브러리를 쓰는 게 안전합니다.

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# 학습된 정책 평가
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

10만 스텝 정도면 LunarLander에서 안정적으로 착륙합니다.

---

## 7. 강화학습이 빛나는 영역

### 7.1 게임

- **DQN (2015)**: 아타리 게임 49종에서 사람 수준 달성
- **알파고 (2016)**: 바둑에서 이세돌 9단 격파
- **알파스타 (2019)**: 스타크래프트 II 그랜드마스터
- **OpenAI Five (2019)**: 도타 2 세계 챔피언팀 격파

### 7.2 로봇 공학

물건 집기, 걷기, 드론 제어처럼 시뮬레이션에서 학습한 정책을 실제 로봇에 옮기는 **sim-to-real**이 활발합니다.

### 7.3 추천 시스템과 광고

광고 노출 순서, 추천 목록 구성을 RL로 최적화. 보상은 클릭률, 체류 시간 등.

### 7.4 LLM 정렬 (RLHF)

ChatGPT, Claude, Gemini의 학습 마지막 단계. 사람이 답변 쌍을 비교해 선호를 표시하면, 그 선호를 보상 모델로 학습하고, PPO로 LLM을 미세조정합니다.

```
1. 사전학습된 LLM
2. 지도 미세조정 (좋은 답변 예시로)
3. 보상 모델 학습 (사람 선호 데이터로)
4. PPO 강화학습 (보상 모델을 보상으로)
```

이 4단계가 ChatGPT 시대를 열었습니다.

---

## 8. 강화학습이 어려운 이유

### 8.1 샘플 효율 낮음

지도학습은 수만 장의 데이터로 학습 가능하지만, RL은 수백만~수억 스텝의 환경 상호작용이 필요합니다. 시뮬레이터 없이는 학습이 거의 불가능한 이유입니다.

### 8.2 보상 설계의 함정

> 모델은 보상을 늘리는 가장 쉬운 방법을 찾습니다. 그게 의도와 다를 수 있습니다.

유명한 사례: 보트 레이스 게임에서 점수만 보상으로 줬더니, 모델이 결승점을 무시하고 점수 부스터를 무한 반복하며 점수를 챙김. 결승점에 못 들어가지만 보상은 최대.

이를 **reward hacking**이라고 합니다. 보상 설계는 강화학습 연구의 절반을 차지합니다.

### 8.3 재현성 어려움

같은 알고리즘, 같은 환경, 다른 시드 → 결과가 천차만별. 학술 논문 결과를 재현하기 매우 어렵습니다.

### 8.4 평가가 까다롭다

지도학습처럼 고정된 검증셋이 없습니다. 평균 보상도 분산이 커서 신뢰하기 어렵습니다.

---

## 9. 시작하는 길

강화학습은 깊이 들어가면 수학과 시스템 양쪽으로 부담이 큽니다. 입문 경로를 추천합니다.

**1단계 — 직관**
- David Silver의 RL 강의 (UCL, YouTube 무료)
- Spinning Up in Deep RL (OpenAI 무료 자료)

**2단계 — 손으로**
- gymnasium으로 CartPole, MountainCar 등 단순 환경 학습
- DQN과 REINFORCE 직접 구현

**3단계 — 실전**
- Stable Baselines3 또는 RLlib으로 PPO/SAC 등 검증 알고리즘 사용
- Atari, MuJoCo, 또는 자기 도메인 환경 적용

**4단계 — 최신 흐름**
- RLHF (LLM 정렬)
- Offline RL (사전 수집 데이터로 학습)
- Multi-Agent RL
- Model-based RL (Dreamer, MuZero)

---

## 10. 한눈에 정리

| 개념 | 한 줄 요약 |
|------|-----------|
| Agent/Env | 학습 주체와 그가 작동하는 세계 |
| State/Action/Reward | 관찰, 선택, 결과 신호 |
| Policy π | 상태에서 행동으로의 함수 |
| Value V, Q | 상태/행동의 장기 가치 |
| 벨만 방정식 | 가치의 재귀적 정의 |
| Q-Learning | 가치 표 업데이트 |
| DQN | Q-함수를 신경망으로 |
| Policy Gradient | 정책을 직접 학습 |
| Actor-Critic | 정책 + 가치 동시 |
| PPO | 안정적 정책 업데이트, 현대 표준 |

**알고리즘 선택 가이드**
- 작은 표 환경 → Q-Learning
- 이산 행동 + 픽셀 입력 → DQN, Rainbow
- 연속 행동 (로봇 제어) → SAC, TD3, PPO
- LLM 정렬, 일반적 첫 선택 → PPO
- 데이터 수집 비용 큼 → Offline RL (CQL, IQL)

---

## 더 깊이 가고 싶으시면

- **「Reinforcement Learning: An Introduction」** by Sutton & Barto — 표준 교과서, 무료 PDF
- **「Spinning Up in Deep RL」** — OpenAI의 입문 자료
- **gymnasium, PettingZoo** — 환경 라이브러리 (단일/멀티 에이전트)
- **Stable Baselines3, CleanRL, RLlib** — 알고리즘 구현체
- **Decision Transformer** — RL을 시퀀스 예측 문제로 재구성
- **MuZero, Dreamer** — Model-based RL의 정점
- **RLHF & DPO** — LLM 정렬 기법

다음 장에서는 마지막 주제로 **AI 윤리와 공정성**을 다룹니다. 모델이 잘 작동하는 것과 모델이 옳게 작동하는 것은 다른 문제입니다.
