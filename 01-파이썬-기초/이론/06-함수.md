# 1.6 함수: 재사용 가능한 부품

## 함수가 왜 필요한가요?

학생 점수의 평균을 구하는 코드를 짠다고 해 봅시다.

```python
scores = [85, 92, 78, 95, 67]

total = 0
for s in scores:
    total += s
average = total / len(scores)
print(average)
```

OK, 잘 돌아가요. 그런데 다른 반의 점수도 평균 내고 싶으면? 다른 학년도? 그때마다 4줄을 복사 붙여넣기 해야 할까요?

이게 바로 **함수가 필요한 순간**입니다.

> **함수(function)** 는 자주 쓰는 코드를 한 번 만들어 두고, **이름만 불러서 재사용**할 수 있게 만든 부품이에요.

```python
def average(scores):
    total = 0
    for s in scores:
        total += s
    return total / len(scores)

print(average([85, 92, 78, 95, 67]))   # 83.4
print(average([90, 88, 76]))           # 84.66...
print(average([100, 100, 100]))        # 100.0
```

한 번 정의해 두면 평생 쓸 수 있어요. 머신러닝 코드의 90%는 함수와 함수의 호출로 이루어집니다. 진짜로요.

---

## 1. 함수 만들기

기본 형태는 이렇습니다.

```python
def 함수이름(매개변수):
    실행할 코드
    return 돌려줄 값
```

예를 들어, 두 수를 더하는 함수.

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)    # 8
```

### 한 줄씩 뜯어보기

```python
def add(a, b):
```
- `def`: "지금부터 함수를 정의할게요" 라는 키워드
- `add`: 함수의 이름 (우리가 정함)
- `(a, b)`: **매개변수 (parameter)** — 함수가 받을 입력값들
- `:`: 콜론 잊지 마세요

```python
    return a + b
```
- 4칸 들여쓰기
- `return`: "이 값을 돌려줄게요"

```python
result = add(3, 5)
```
- 함수 **호출 (call)** — `함수이름(인자1, 인자2)`
- `3`과 `5`가 함수 안의 `a`, `b`에 들어감
- 함수가 돌려준 값(`8`)이 `result`에 담김

---

## 2. 매개변수와 인자

용어를 정리하고 갈게요. (헷갈려하시는 분들이 많아요.)

```python
def add(a, b):    # ← a, b는 "매개변수 (parameter)"
    return a + b

add(3, 5)         # ← 3, 5는 "인자 (argument)"
```

엄밀히는 다른 단어지만, 일상에서는 거의 섞어 써요. 그냥 같다고 보셔도 무방합니다.

### 매개변수 개수

원하는 만큼 가질 수 있어요.

```python
def greet():
    print("안녕하세요")

def square(x):
    return x * x

def calculate(a, b, c, d):
    return (a + b) * (c - d)
```

### 매개변수 기본값

매개변수에 기본값을 줄 수 있어요. 그러면 호출할 때 안 줘도 됩니다.

```python
def greet(name="손님"):
    print(f"{name}님 안녕하세요")

greet()           # 손님님 안녕하세요
greet("철수")     # 철수님 안녕하세요
```

머신러닝 함수에서 정말 자주 봅니다. 예를 들어 `train_test_split(X, y, test_size=0.2)`에서 `test_size=0.2`는 "안 주면 0.2로 한다"는 뜻이에요.

### 키워드 인자 (keyword argument)

호출할 때 매개변수 이름을 명시할 수도 있어요.

```python
def order(menu, size, drink):
    print(f"{menu} {size}, {drink}")

# 위치 순서대로
order("피자", "L", "콜라")

# 키워드로 (순서 바뀌어도 됨)
order(drink="콜라", menu="피자", size="L")

# 섞어서도 됨 (위치 인자가 먼저)
order("피자", drink="콜라", size="L")
```

매개변수가 많아질 때, 키워드로 명시하면 코드 읽기가 훨씬 쉬워져요.

```python
# ❌ 무슨 뜻이지?
model = SVC(1.0, "rbf", 3, 0.001, True)

# ✅ 명확함
model = SVC(C=1.0, kernel="rbf", degree=3, tol=0.001, shrinking=True)
```

머신러닝 코드는 거의 다 이렇게 키워드 인자로 적어요.

---

## 3. return: 돌려주기

`return`은 함수의 결과를 호출한 곳으로 돌려보내는 명령이에요.

```python
def double(x):
    return x * 2

result = double(5)
print(result)    # 10
```

### return이 없으면?

함수 안에 `return`이 없거나, 그냥 `return`만 적으면, **`None`을 돌려줍니다.**

```python
def greet(name):
    print(f"{name}님 안녕하세요")
    # return 없음

result = greet("철수")
# 철수님 안녕하세요 (출력)

print(result)    # None
```

이게 헷갈릴 수 있는 한 가지예요. **"화면에 출력하는 것"과 "값을 돌려주는 것"은 다릅니다.**

```python
def square_print(x):
    print(x * x)        # 화면에 출력만 함

def square_return(x):
    return x * x         # 값을 돌려줌

a = square_print(5)     # 화면에 25 보임
b = square_return(5)    # 화면에는 안 보임

print("a =", a)         # a = None
print("b =", b)         # b = 25

# return이 있어야 다른 계산에 쓸 수 있음
print(square_return(5) + 100)   # 125 OK
print(square_print(5) + 100)    # ❌ TypeError!
```

### return은 함수를 즉시 끝냅니다

`return`을 만나면 함수 실행이 거기서 끝나요. 아래 코드는 실행 안 됩니다.

```python
def check_age(age):
    if age < 0:
        return "잘못된 나이"
    
    if age < 18:
        return "미성년자"
    
    return "성인"
    
    print("이 줄은 영원히 실행 안 됨")    # 절대 실행 X
```

이걸 잘 활용하면 if-elif-else를 깔끔하게 정리할 수 있어요.

### 여러 값 돌려주기

`return` 다음에 콤마로 여러 값을 적으면 튜플로 돌려줘요.

```python
def get_min_max(numbers):
    return min(numbers), max(numbers)

mn, mx = get_min_max([3, 7, 1, 9, 4])
print(f"최솟값 {mn}, 최댓값 {mx}")
```

---

## 4. 함수의 진짜 가치

함수의 진짜 가치는 단순히 "코드 재사용"이 아니에요. 더 중요한 게 있어요.

### 1) 이름이 곧 설명이 됩니다

```python
# ❌ 함수 없이
total = 0
for s in scores:
    total += s
average_score = total / len(scores)

if average_score >= 60:
    print("합격")
```

```python
# ✅ 함수로
average_score = average(scores)

if average_score >= 60:
    print("합격")
```

두 번째 코드가 훨씬 읽기 쉬워요. **`average(scores)`** 라는 한 단어가 4줄을 대체합니다. **코드는 컴퓨터가 읽는 게 아니라 사람이 읽는 거예요.** 이걸 명심하세요.

### 2) 한 곳만 고치면 됩니다

평균 계산하는 코드를 한 100군데에서 쓰고 있는데, "평균이 아니라 중앙값으로 바꿔야겠다"고 결정됐다고 해 봅시다.

함수를 안 썼다면? 100군데를 다 고쳐야 해요.
함수를 썼다면? 함수 정의 하나만 고치면 돼요.

### 3) 테스트가 쉽습니다

함수는 입력을 주면 출력이 나오는 작은 단위라서, **하나하나 따로 검증**할 수 있어요.

```python
assert average([10, 20, 30]) == 20    # 통과
assert average([100]) == 100          # 통과
```

`assert`는 "이게 참이 아니면 에러를 내라"는 명령이에요. 테스트할 때 자주 씁니다.

---

## 5. 함수 안의 함수, 함수가 변수

이건 좀 응용이지만, 머신러닝 코드에서 보게 되실 패턴이에요.

### 함수도 변수처럼 다룰 수 있어요

```python
def greet(name):
    return f"안녕 {name}"

# 함수를 변수에 담음
my_greet = greet
print(my_greet("철수"))    # 안녕 철수
```

### 함수를 함수의 인자로 넘길 수 있어요

```python
def apply_twice(func, value):
    return func(func(value))

def add_one(x):
    return x + 1

print(apply_twice(add_one, 5))    # 7   (5 → 6 → 7)
```

머신러닝에서 이런 패턴을 콜백(callback)이라고 부르며 자주 씁니다. **PyTorch의 옵티마이저, 활성화 함수**가 다 이런 식으로 함수를 받아요.

### lambda: 한 줄짜리 익명 함수

```python
# 정식 함수
def double(x):
    return x * 2

# 같은 걸 lambda로
double = lambda x: x * 2

print(double(5))    # 10
```

`lambda 매개변수: 표현식` 형태예요. **이름 없이 그 자리에서만 쓸 함수**를 만들 때 편해요.

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_desc = sorted(numbers, key=lambda x: -x)
print(sorted_desc)    # [9, 6, 5, 4, 3, 2, 1, 1]
```

다 이해 못 하셔도 괜찮아요. **"아, 이런 게 있구나"** 정도만요.

---

## 6. docstring: 함수 설명 적기

함수 첫 줄 바로 아래에 따옴표 세 개로 설명을 적을 수 있어요. 이걸 **docstring**이라고 합니다.

```python
def average(scores):
    """학생들의 점수 리스트를 받아 평균을 돌려준다.
    
    인자:
        scores: 숫자들의 리스트
    
    반환:
        평균값 (float)
    """
    return sum(scores) / len(scores)
```

이게 왜 좋냐면, **자동 도움말**이 됩니다.

```python
help(average)
# Help on function average in module __main__:
#
# average(scores)
#     학생들의 점수 리스트를 받아 평균을 돌려준다.
#     ...
```

머신러닝 라이브러리들은 거의 다 docstring이 잘 되어 있어요. 그래서 우리가 `help(SVC)` 하면 사용법이 나오는 거예요.

---

## 정리: 한 페이지 요약

```python
# 정의
def add(a, b):
    return a + b

# 호출
add(3, 5)              # 8

# 기본값
def greet(name="손님"):
    return f"{name}님 안녕"

greet()                # "손님님 안녕"
greet("철수")          # "철수님 안녕"

# 키워드 인자
def order(menu, size, drink):
    ...

order("피자", drink="콜라", size="L")

# 여러 값 반환
def min_max(nums):
    return min(nums), max(nums)

mn, mx = min_max([1, 5, 3])

# docstring
def add(a, b):
    """두 수를 더한다."""
    return a + b
```

---

## 자주 묻는 질문

> **Q. 함수 안에서 만든 변수를 밖에서 쓸 수 있어요?**
>
> 못 씁니다. 함수 안에서 만든 변수는 함수 안에서만 살아 있어요(이걸 **지역 변수, local variable**라고 합니다).
> ```python
> def foo():
>     x = 10
>
> foo()
> print(x)   # ❌ NameError
> ```
> 함수 밖으로 값을 가져가고 싶으면 `return`을 쓰세요.

> **Q. 함수가 다른 함수를 부를 수 있나요?**
>
> 네, 당연히 됩니다. 자주 그렇게 합니다.
> ```python
> def square(x):
>     return x * x
>
> def sum_of_squares(a, b):
>     return square(a) + square(b)
>
> print(sum_of_squares(3, 4))    # 9 + 16 = 25
> ```

> **Q. 함수 이름은 어떻게 짓는 게 좋아요?**
>
> 보통 **동사**로 짓습니다. 그게 자연스러워요.
> - `calculate_average` ✅ (계산한다)
> - `get_user_name` ✅ (가져온다)
> - `is_valid` ✅ (~인지 묻는다, bool 반환 함수)
> - `average` ⚠️ (명사 — 가능하지만 약간 어색)
>
> 그리고 **소문자에 단어 사이는 `_`**(언더스코어)로 잇는 게 파이썬 관례예요. 이걸 **snake_case**라고 합니다.

➡️ 다음: [1.7 실전 예제: 계산기 만들기](07-실전-예제.md)
