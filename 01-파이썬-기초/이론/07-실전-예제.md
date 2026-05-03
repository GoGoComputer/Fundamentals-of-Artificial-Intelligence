# 1.7 실전 예제: 계산기 만들기

## 지금까지 배운 모든 걸 합쳐 봅시다

이번 글은 새로운 개념이 거의 없어요. 대신, 1장에서 배운 모든 걸 합쳐서 **진짜 동작하는 작은 프로그램**을 함께 만들어 봅니다.

만들 것: **간단한 계산기**

목표:
- 사용자에게 두 숫자와 연산을 입력받는다
- 결과를 보여준다
- 0으로 나누기 같은 잘못된 입력을 막는다
- 사용자가 끝낼 때까지 반복한다

---

## 1단계: 가장 단순한 버전

먼저 한 번만 동작하는 가장 단순한 계산기부터.

```python
a = 10
b = 3
op = "+"

if op == "+":
    result = a + b
elif op == "-":
    result = a - b
elif op == "*":
    result = a * b
elif op == "/":
    result = a / b
else:
    result = None

print(f"{a} {op} {b} = {result}")
```

```
10 + 3 = 13
```

OK, 동작해요. 그런데 이걸 매번 코드를 고쳐야 새로운 계산을 할 수 있어요. 함수로 만들어 봅시다.

---

## 2단계: 함수로 만들기

```python
def calculate(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    else:
        return None


# 써 보기
print(calculate(10, 3, "+"))    # 13
print(calculate(10, 3, "-"))    # 7
print(calculate(10, 3, "*"))    # 30
print(calculate(10, 3, "/"))    # 3.333...
print(calculate(10, 3, "?"))    # None
```

훨씬 깔끔해졌어요. 함수 하나로 모든 계산이 가능해요.

---

## 3단계: 0으로 나누기 막기

`10 / 0` 은 어떻게 될까요?

```python
print(calculate(10, 0, "/"))
# ZeroDivisionError !
```

에러가 나요. 이런 입력에도 우아하게 대응해야 합니다.

```python
def calculate(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            return None    # 0으로 나눌 수 없음
        return a / b
    else:
        return None


result = calculate(10, 0, "/")
if result is None:
    print("오류: 잘못된 계산입니다")
else:
    print(f"결과: {result}")
```

```
오류: 잘못된 계산입니다
```

⚠️ **`None` 비교는 `==` 가 아니라 `is`를 쓰세요.** 이건 파이썬의 관례예요. `if result is None:` 이렇게요.

---

## 4단계: 사용자 입력 받기

`input()` 함수를 쓰면 사용자에게 입력을 받을 수 있어요.

```python
text = input("숫자를 입력하세요: ")
print(f"입력값: {text}")
print(f"자료형: {type(text)}")
```

⚠️ **`input()`은 항상 문자열(str)을 돌려줍니다.** 숫자 계산을 하려면 `int()` 또는 `float()`로 변환해야 해요.

```python
a_text = input("첫 번째 숫자: ")
b_text = input("두 번째 숫자: ")
op = input("연산 (+, -, *, /): ")

a = float(a_text)    # 문자 → 숫자
b = float(b_text)

result = calculate(a, b, op)
if result is None:
    print("계산할 수 없습니다")
else:
    print(f"{a} {op} {b} = {result}")
```

---

## 5단계: 반복하기

한 번 계산하고 끝나면 심심하니까, 사용자가 종료할 때까지 계속 돌게 만들어 봅시다.

```python
def calculate(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            return None
        return a / b
    else:
        return None


print("=" * 30)
print("  파이썬 계산기")
print("=" * 30)

while True:
    a_text = input("\n첫 번째 숫자 (q로 종료): ")
    if a_text == "q":
        break
    
    b_text = input("두 번째 숫자: ")
    op = input("연산 (+, -, *, /): ")
    
    a = float(a_text)
    b = float(b_text)
    
    result = calculate(a, b, op)
    if result is None:
        print("→ 계산할 수 없습니다")
    else:
        print(f"→ {a} {op} {b} = {result}")

print("\n계산기를 종료합니다. 안녕히가세요!")
```

이걸 Colab에 그대로 붙여넣고 실행해 보세요. 진짜 계산기처럼 동작합니다.

---

## 6단계: 더 나아가기 (선택)

여기서 멈춰도 충분합니다. 하지만 좀 더 욕심을 부려본다면, 이런 기능들을 추가해 볼 수 있어요.

### 6-1. 잘못된 입력 처리

지금 코드는 사용자가 숫자가 아닌 글자를 입력하면 죽어요.

```python
첫 번째 숫자: abc    # ValueError!
```

이걸 막으려면 `try / except`라는 걸 쓰는데, 1장에서 안 다뤘으니 미리보기 정도로만:

```python
try:
    a = float(a_text)
    b = float(b_text)
except ValueError:
    print("→ 숫자를 입력해 주세요")
    continue
```

`try` 블록을 시도해 보고, 에러가 나면 `except` 블록으로 가는 식이에요. 자세한 건 더 깊이 파이썬을 배우실 때 보세요.

### 6-2. 계산 기록 남기기

지금까지의 계산을 모두 기억하게 해 봅시다.

```python
history = []   # 빈 리스트

while True:
    # ... 계산 코드 ...
    
    if result is not None:
        history.append(f"{a} {op} {b} = {result}")

# 끝날 때 기록 출력
print("\n[계산 기록]")
for record in history:
    print(f"  - {record}")
```

### 6-3. 더 많은 연산

거듭제곱이나 나머지도 추가해 봅시다.

```python
def calculate(a, b, op):
    if op == "+": return a + b
    elif op == "-": return a - b
    elif op == "*": return a * b
    elif op == "/":
        if b == 0: return None
        return a / b
    elif op == "**": return a ** b      # 거듭제곱
    elif op == "%":
        if b == 0: return None
        return a % b                     # 나머지
    else:
        return None
```

---

## 완성된 최종 코드

위 모든 기능을 다 합친 코드는 [실습/05_계산기.py](../실습/05_계산기.py)에 있습니다.

지금 그 파일을 한 번 보세요. **모르는 줄이 거의 없으셔야 합니다.** 모르는 줄이 있다면, 해당 글로 돌아가서 다시 보세요.

만약 거의 다 이해되신다면, **축하드립니다. 1장 끝났습니다.** 진짜로요.

---

## 1장을 마치며

지금까지 다룬 내용:

- ✅ `print`, 변수, 주석, 들여쓰기, `import`
- ✅ 자료형 (int, float, str, bool, None)
- ✅ 자료구조 (list, tuple, set, dict)
- ✅ 연산자 (산술, 비교, 논리, 할당)
- ✅ 제어문 (if, for, while, break, continue)
- ✅ 함수 (def, return, 매개변수, 기본값)

이게 다음 장 머신러닝 코드를 읽는 데 필요한 파이썬의 **거의 전부**입니다. 진짜로요. 이 정도면 충분해요.

물론 파이썬에는 더 많은 기능이 있어요. 클래스, 예외처리, 모듈 만들기, 데코레이터, 비동기, 컨텍스트 매니저 등등. 하지만 그건 **필요할 때 배우는 게 가장 효율적입니다.** 미리 모든 걸 배우려고 하면 흥미가 떨어져요. 머신러닝을 하다가 "아, 이런 게 필요하네" 싶을 때 그때그때 배우세요.

---

## 자가 진단 테스트

다음 코드들을 보고 결과를 예상해 보세요. 실행해 보지 않고요.
다 맞히시면 1장은 졸업입니다. 못 맞히시는 게 있으시면 해당 부분으로 돌아가세요.

```python
# 1
x = 10
y = "10"
print(x + int(y))
# ?

# 2
fruits = ["사과", "바나나", "딸기"]
print(fruits[-2])
# ?

# 3
person = {"name": "철수", "age": 25}
print(person.get("address", "없음"))
# ?

# 4
for i in range(2, 8, 2):
    print(i, end=" ")
# ?

# 5
def f(x, y=10):
    return x + y

print(f(5))
print(f(5, 20))
# ?

# 6
nums = [3, 1, 4, 1, 5]
print(sum(set(nums)))
# ?
```

<details>
<summary>정답 (펼쳐 보기)</summary>

1. `20` (10 + 10)
2. `바나나` (뒤에서 두 번째)
3. `없음` (키가 없으면 기본값)
4. `2 4 6` (8은 포함 X)
5. `15` 와 `25`
6. `13` (set으로 중복 제거: {1, 3, 4, 5} → 합 13)

</details>

---

다 맞히셨나요? 그럼 진짜 본론으로 들어갑시다.

➡️ **[2장. 머신러닝 — 분류](../../02-머신러닝-분류/)**
