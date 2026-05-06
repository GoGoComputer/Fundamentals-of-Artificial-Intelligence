# 1.5 제어문: 조건과 반복

## 제어문이 왜 필요한가요?

지금까지의 코드는 **위에서 아래로 한 줄씩** 실행됐어요.

```python
print("첫째 줄")
print("둘째 줄")
print("셋째 줄")
```

그런데 실제 프로그램은 그렇게 단순하지 않잖아요.

- "사용자 나이가 19세 이상이면 입장시키고, 아니면 막아라"
- "학생 100명의 점수를 모두 더해라"
- "정답을 맞힐 때까지 계속 물어봐라"

이런 걸 표현하려면 **흐름을 바꾸는 도구**가 필요합니다. 그게 제어문이에요. 두 가지가 있어요.

1. **조건문 (if)** — "이러면 이걸 하고, 저러면 저걸 해라"
2. **반복문 (for, while)** — "여러 번 반복해라"

---

## 1. 조건문: if / elif / else

### 가장 단순한 if

```python
age = 25

if age >= 19:
    print("성인입니다")
```

**`if 조건:` 다음에 들여쓰기로 묶인 줄**들은, 조건이 `True`일 때만 실행돼요. `False`면 그냥 건너뜁니다.

⚠️ **중요한 두 가지:**
1. `if` 다음에 **콜론(`:`)** 을 꼭 붙여야 해요.
2. 다음 줄은 **공백 4칸 들여쓰기**.

### if-else: 둘 중 하나

조건이 `True`면 첫 번째, `False`면 두 번째 블록을 실행해요.

```python
age = 15

if age >= 19:
    print("성인입니다")
else:
    print("미성년자입니다")
```

### if-elif-else: 여러 갈래

조건이 여러 개면 `elif`로 이어 붙여요. (else if의 줄임)

```python
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("F")
```

위에서부터 차례로 검사해서, **처음 만나는 True에서 실행하고 나머지는 무시**합니다. 위 코드는 `score=85`니까 두 번째 조건(`>= 80`)에서 멈춰서 `B`를 출력해요.

### 중첩된 if

if 안에 또 if를 넣을 수 있어요.

```python
age = 25
has_license = True

if age >= 18:
    if has_license:
        print("운전 가능")
    else:
        print("나이는 됐지만 면허가 없음")
else:
    print("나이가 안 됨")
```

물론 이 경우는 `and`로 한 줄로 쓸 수도 있어요.

```python
if age >= 18 and has_license:
    print("운전 가능")
```

너무 많이 중첩하면 읽기가 어려워져요. **3단계 이상은 잘 짜였다고 보기 어렵다**는 게 일반적인 의견입니다.

### 조건이 "참이냐"의 다양한 모습

`if 조건:`에서 조건이 꼭 비교 연산이어야 하는 건 아니에요. 어떤 값이든 와도 됩니다.

```python
name = "철수"
if name:    # 빈 문자열이 아니면 True
    print(f"{name}님 안녕하세요")

numbers = []
if numbers:    # 빈 리스트는 False
    print("리스트에 뭐가 있어요")
else:
    print("리스트가 비어있어요")    # ← 이거 출력됨
```

파이썬에서 다음 값들은 **거짓(False)** 으로 취급돼요.

| 거짓으로 취급되는 값 |
|---|
| `False` |
| `None` |
| `0`, `0.0` |
| `""` (빈 문자열) |
| `[]` (빈 리스트) |
| `()` (빈 튜플) |
| `{}` (빈 딕셔너리) |
| `set()` (빈 셋) |

이 외엔 다 **참(True)**입니다. 빈 컨테이너를 검사할 때 정말 자주 씁니다.

```python
# ❌ 길게
if len(numbers) > 0:
    ...

# ✅ 짧게
if numbers:
    ...
```

---

## 2. for문: 반복하기

`for`는 **"이 안의 것들을 하나씩 꺼내서 반복하라"** 라는 뜻입니다.

### 리스트 순회

```python
fruits = ["사과", "바나나", "딸기"]

for fruit in fruits:
    print(fruit)
```

```
사과
바나나
딸기
```

`fruit`는 매번 새로운 값이 들어오는 변수예요. 첫 번째 반복에선 `"사과"`, 두 번째 `"바나나"`, 세 번째 `"딸기"`.

이름을 `fruit`이라고 한 건 그냥 우리가 정한 이름일 뿐이에요. `f`, `item`, 아무거나 다 되지만 의미 있는 이름을 쓰는 게 좋아요.

### range로 숫자 반복

`range(n)`은 `0`부터 `n-1`까지의 숫자를 만들어줘요.

```python
for i in range(5):
    print(i)
```

```
0
1
2
3
4
```

⚠️ `range(5)`는 0, 1, 2, 3, 4입니다. **5는 포함 안 돼요.** (리스트 슬라이싱이랑 같은 규칙이에요.)

### range의 다양한 사용법

```python
range(5)         # 0, 1, 2, 3, 4
range(2, 7)      # 2, 3, 4, 5, 6
range(0, 10, 2)  # 0, 2, 4, 6, 8  (2칸씩)
range(10, 0, -1) # 10, 9, 8, ..., 1  (역순)
```

3개의 인자: `range(시작, 끝, 단계)`. 끝은 항상 포함 안 됨.

### enumerate: 인덱스도 같이

종종 "지금 몇 번째인가?"가 필요할 때가 있어요.

```python
fruits = ["사과", "바나나", "딸기"]

for i, fruit in enumerate(fruits):
    print(f"{i}번째: {fruit}")
```

```
0번째: 사과
1번째: 바나나
2번째: 딸기
```

`enumerate`는 `(인덱스, 값)` 튜플을 만들어줘요. 진짜 자주 씁니다.

### 딕셔너리 순회

```python
person = {"name": "철수", "age": 25}

# 키만
for key in person:
    print(key)

# 값만
for value in person.values():
    print(value)

# 키와 값 둘 다
for key, value in person.items():
    print(f"{key} = {value}")
```

머신러닝에서 모델 설정값들을 딕셔너리로 만들고 출력할 때 마지막 패턴 자주 씁니다.

```python
config = {"learning_rate": 0.001, "epochs": 10, "batch_size": 32}

for k, v in config.items():
    print(f"{k}: {v}")
```

### 누적 합계 패턴

이 패턴은 정말 정말 자주 봅니다. 외워두세요.

```python
numbers = [10, 20, 30, 40, 50]

total = 0
for n in numbers:
    total += n

print(total)    # 150
```

머신러닝에서 한 epoch의 loss를 모두 더해서 평균 내는 패턴이 이거예요.

```python
total_loss = 0
for batch in batches:
    loss = model_compute_loss(batch)
    total_loss += loss

avg_loss = total_loss / len(batches)
```

### 짧게 쓰기: 내장 함수

사실 자주 쓰는 패턴들은 파이썬에 이미 함수로 있어요.

```python
numbers = [10, 20, 30, 40, 50]

print(sum(numbers))    # 150  (모두 더하기)
print(max(numbers))    # 50
print(min(numbers))    # 10
print(len(numbers))    # 5
```

이런 거 있는데 굳이 for문을 직접 쓸 필요는 없어요. **있는 거 쓰세요.**

---

## 3. while문: 조건이 참인 동안

`for`는 "정해진 횟수만큼" 반복할 때, `while`은 **"조건이 참인 동안" 계속 반복**할 때 씁니다.

```python
count = 0

while count < 5:
    print(count)
    count += 1
```

```
0
1
2
3
4
```

⚠️ **위험한 함정: 무한 루프**

`while`을 쓰실 땐 **반드시 조건을 거짓으로 만들 코드**가 안에 있어야 해요. 안 그러면 영원히 안 끝납니다.

```python
count = 0
while count < 5:
    print(count)
    # count += 1 을 깜빡 잊으면...
    # 0이 영원히 출력됨!
```

Colab에서 무한 루프가 돌면 셀 옆의 ▶ 버튼이 ⏹로 바뀌어요. 그걸 눌러서 멈출 수 있어요.

### while을 언제 써요?

대부분의 경우 `for`로 충분합니다. `while`은 **"몇 번 반복할지 미리 모를 때"** 써요.

```python
# 사용자가 q를 입력할 때까지 받기
while True:
    answer = input("뭐든 입력하세요 (q로 종료): ")
    if answer == "q":
        break
    print(f"입력: {answer}")
```

`while True:`는 영원한 루프지만, `break`로 빠져나옵니다. 자주 쓰는 패턴이에요.

---

## 4. break와 continue

반복문 안에서 흐름을 조절하는 두 가지가 있어요.

### break: 반복 멈추기

```python
for n in range(100):
    if n == 5:
        break
    print(n)
```

```
0
1
2
3
4
```

`n == 5`가 되면 즉시 for문 전체에서 빠져나옵니다.

### continue: 다음 반복으로

```python
for n in range(10):
    if n % 2 == 0:    # 짝수면
        continue       # 이번 반복 건너뛰고 다음으로
    print(n)
```

```
1
3
5
7
9
```

홀수만 출력돼요. `continue`는 "이번 한 번만 건너뛰기"입니다.

---

## 5. 합쳐서 써보기

지금까지 배운 걸 다 써서 학생 점수를 처리해 봅시다.

```python
students = {
    "철수": 85,
    "영희": 92,
    "민수": 78,
    "수진": 95,
    "지호": 67,
}

# 1. 모든 점수 출력
print("[전체 학생]")
for name, score in students.items():
    print(f"  {name}: {score}점")

# 2. 평균 계산
total = 0
for score in students.values():
    total += score
average = total / len(students)
print(f"\n평균: {average:.1f}점")

# 3. 합격자만 출력 (80점 이상)
print("\n[합격자]")
for name, score in students.items():
    if score >= 80:
        print(f"  {name}: {score}점")

# 4. 최고점 학생
best_name = ""
best_score = 0
for name, score in students.items():
    if score > best_score:
        best_score = score
        best_name = name
print(f"\n최고점: {best_name} ({best_score}점)")
```

직접 실행해 보세요. 결과는 이래요.

```
[전체 학생]
  철수: 85점
  영희: 92점
  민수: 78점
  수진: 95점
  지호: 67점

평균: 83.4점

[합격자]
  철수: 85점
  영희: 92점
  수진: 95점

최고점: 수진 (95점)
```

이런 코드를 술술 짜실 수 있게 되시면, 1장의 80%는 끝나신 겁니다.

---

## 보너스: 리스트 컴프리헨션 (List Comprehension)

처음 배우실 땐 어색하지만, 파이썬을 좀 더 쓰시다 보면 만나게 되는 예쁜 표현이에요.

```python
# for문으로 짝수만 모은 리스트 만들기
evens = []
for n in range(10):
    if n % 2 == 0:
        evens.append(n)
print(evens)    # [0, 2, 4, 6, 8]

# 리스트 컴프리헨션으로 한 줄에
evens = [n for n in range(10) if n % 2 == 0]
print(evens)    # [0, 2, 4, 6, 8]
```

이 한 줄이 위 4줄과 똑같은 역할을 합니다. 처음에는 쓰지 않으셔도 돼요. **다른 사람의 코드를 읽을 때 이게 무슨 뜻인지만 아시면 됩니다.**

```python
# 패턴: [표현식 for 변수 in 반복 if 조건]
squares = [n * n for n in range(5)]
# [0, 1, 4, 9, 16]

names = ["철수", "영희", "민수"]
upper = [name.upper() for name in names]
# 각 이름을 대문자로 (영어일 경우)
```

머신러닝 코드에서 자주 보이게 됩니다.

---

## 정리: 한 페이지 요약

```python
# 조건문
if x > 0:
    print("양수")
elif x < 0:
    print("음수")
else:
    print("0")

# for문 (가장 자주 씀)
for item in [1, 2, 3]:
    print(item)

for i in range(5):    # 0, 1, 2, 3, 4
    print(i)

for i, name in enumerate(names):
    print(f"{i}: {name}")

for k, v in dictionary.items():
    print(k, v)

# while문 (덜 자주 씀)
while condition:
    do_something()

# 흐름 조절
break       # 반복 즉시 종료
continue    # 다음 반복으로
```

---

## 자주 묻는 질문

> **Q. for랑 while 중 어느 걸 써야 해요?**
>
> 거의 항상 `for`입니다. 다음 두 경우에만 `while`:
> 1. 몇 번 반복할지 미리 모를 때 (사용자 입력 대기 등)
> 2. 어떤 조건이 만족될 때까지 (수렴할 때까지 등)

> **Q. 들여쓰기가 너무 깊어지면 어떡해요?**
>
> 일반적으로 **3단계 이상이면 코드를 다시 짜라는 신호**입니다. 함수로 빼거나, 조건을 합치거나 (`and`, `or`), `continue`로 일찍 빠져나오는 식으로 평탄하게 만드세요.
> ```python
> # 안 좋음
> for x in items:
>     if condition1:
>         if condition2:
>             if condition3:
>                 do_something()
>
> # 좋음
> for x in items:
>     if not condition1: continue
>     if not condition2: continue
>     if not condition3: continue
>     do_something()
> ```

> **Q. for문 안에서 리스트를 수정하면 안 된다는 말을 들었어요.**
>
> 맞아요. 위험합니다. 수정해야 한다면 새 리스트를 만들어서 거기에 담으세요.
> ```python
> # ❌ 위험
> for item in items:
>     if condition: items.remove(item)
>
> # ✅ 안전
> new_items = [item for item in items if not condition]
> ```

➡️ 다음: [1.6 함수: 재사용 가능한 부품](06-함수.md)
