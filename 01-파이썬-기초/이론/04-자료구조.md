# 1.4 자료구조: 데이터를 담는 그릇 4종

## 왜 자료구조가 필요한가요?

지금까지는 변수 하나에 값 하나를 담았어요.

```python
student1 = "철수"
student2 = "영희"
student3 = "민수"
```

학생이 100명이면 어떡하죠? 변수를 100개 만드는 건 말이 안 되겠죠.

**자료구조(data structure)는 여러 값을 하나로 묶어서 담는 그릇**이에요. 파이썬의 기본 자료구조 4종을 비유로 보여드릴게요.

| 그릇 | 비유 | 특징 |
|------|------|------|
| **리스트 (list)** | 줄 서 있는 사람들 | 순서 있음, 중복 OK, 바꿀 수 있음 |
| **튜플 (tuple)** | 자물쇠 채워진 줄 | 순서 있음, 중복 OK, 바꿀 수 없음 |
| **셋 (set)** | 봉지 안의 구슬들 | 순서 없음, 중복 X, 바꿀 수 있음 |
| **딕셔너리 (dict)** | 사전 (단어→뜻) | 키-값 쌍, 키 중복 X |

이 네 가지를 자유자재로 다루실 수 있게 되시면 파이썬 자료구조는 80% 정복하신 겁니다.

---

## 1. 리스트 (list)

가장 기본적이고 가장 많이 쓰는 자료구조예요.

### 만들기

대괄호 `[ ]` 안에 콤마로 구분해서 넣으면 됩니다.

```python
fruits = ["사과", "바나나", "딸기"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]   # 다른 자료형 섞어도 OK
empty = []                           # 빈 리스트

print(fruits)    # ['사과', '바나나', '딸기']
```

### 인덱싱: 한 개 꺼내기

리스트 안의 값들에는 **번호(인덱스)** 가 매겨져 있어요. **0부터 시작합니다.**

```python
fruits = ["사과", "바나나", "딸기"]
#         [0]      [1]      [2]

print(fruits[0])    # 사과
print(fruits[1])    # 바나나
print(fruits[2])    # 딸기
```

⚠️ **꼭 기억하세요: 첫 번째는 `[1]`이 아니라 `[0]`입니다.** 처음 배우실 때 가장 자주 실수하는 부분이에요.

### 음수 인덱스: 뒤에서부터

마지막 원소를 가져오고 싶을 땐 `-1`을 쓰면 돼요.

```python
fruits = ["사과", "바나나", "딸기"]

print(fruits[-1])    # 딸기  (마지막)
print(fruits[-2])    # 바나나 (뒤에서 두 번째)
print(fruits[-3])    # 사과   (뒤에서 세 번째 = 첫 번째)
```

리스트 길이를 모를 때 마지막 원소를 가져오는 데 정말 편해요.

### 슬라이싱: 여러 개 꺼내기

`[시작:끝]` 형태로 여러 원소를 한 번에 가져올 수 있어요.

```python
numbers = [10, 20, 30, 40, 50]

print(numbers[1:4])    # [20, 30, 40]   인덱스 1, 2, 3
print(numbers[:3])     # [10, 20, 30]   처음부터 인덱스 2까지
print(numbers[2:])     # [30, 40, 50]   인덱스 2부터 끝까지
print(numbers[:])      # [10, 20, 30, 40, 50]  전체 복사
```

⚠️ **헷갈리는 한 가지: 끝 인덱스는 포함되지 않아요.** `numbers[1:4]`는 인덱스 1, 2, 3까지 (4는 빠짐). 이걸 "open interval"이라고 부르는데, 이유는 길이 계산이 편해서예요.

```python
length = 4 - 1   # 3개  ← 슬라이싱 길이를 이렇게 계산할 수 있음
```

### 단계 (step)도 줄 수 있어요

```python
numbers = [10, 20, 30, 40, 50, 60, 70, 80]

print(numbers[::2])    # [10, 30, 50, 70]   2칸씩 건너뜀
print(numbers[::-1])   # [80, 70, 60, ...]   역순
```

`numbers[::-1]`은 리스트를 뒤집는 가장 짧은 방법이에요. 알아두면 자주 씁니다.

### 값 바꾸기

리스트는 **변경 가능(mutable)** 해요. 인덱스로 접근해서 바꿀 수 있어요.

```python
fruits = ["사과", "바나나", "딸기"]
fruits[1] = "포도"
print(fruits)    # ['사과', '포도', '딸기']
```

### 자주 쓰는 메소드들

리스트는 여러 가지 동작을 할 수 있어요. 점 `.`을 붙여서 호출합니다.

```python
fruits = ["사과", "바나나"]

# 끝에 추가
fruits.append("딸기")
print(fruits)    # ['사과', '바나나', '딸기']

# 여러 개 한꺼번에 추가
fruits.extend(["포도", "수박"])
print(fruits)    # ['사과', '바나나', '딸기', '포도', '수박']

# 특정 위치에 삽입
fruits.insert(0, "오렌지")
print(fruits)    # ['오렌지', '사과', '바나나', ...]

# 값으로 삭제 (첫 번째 만나는 것)
fruits.remove("바나나")
print(fruits)

# 인덱스로 삭제 (그리고 그 값을 반환)
last = fruits.pop()
print(last)    # 수박
print(fruits)

# 길이 확인
print(len(fruits))

# 정렬
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()
print(numbers)    # [1, 1, 2, 3, 4, 5, 6, 9]

numbers.sort(reverse=True)
print(numbers)    # [9, 6, 5, 4, 3, 2, 1, 1]

# 뒤집기
numbers.reverse()
print(numbers)
```

다 외우려 하지 마세요. 자주 쓸수록 자연스럽게 외워집니다.

### 리스트 안의 리스트 (2차원)

리스트 안에 리스트를 넣을 수 있어요. 표를 표현할 때 자주 씁니다.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

print(matrix[0])       # [1, 2, 3]   (첫 번째 행)
print(matrix[0][1])    # 2           (첫 번째 행의 두 번째 값)
print(matrix[2][2])    # 9           (마지막)
```

머신러닝에서 이미지(픽셀의 2차원 배열)나 표 데이터를 표현할 때 이런 식으로 씁니다. 다만 진짜 머신러닝에서는 그냥 리스트 대신 **NumPy 배열**을 써요. 그건 2장에서 만납니다.

---

## 2. 튜플 (tuple)

리스트랑 거의 똑같은데, **한 번 만들면 바꿀 수 없어요.**

```python
point = (3, 5)        # 소괄호 ( )
colors = ("빨강", "초록", "파랑")

print(point[0])       # 3
print(colors[1])      # 초록

colors[0] = "노랑"    # ❌ TypeError !
```

### 왜 못 바꾸는 게 좋은 거예요?

처음 들으면 "리스트가 더 좋은 거 아니야?" 싶죠. 그런데 못 바꿔서 좋은 경우가 있어요.

1. **실수로 바꿀 일이 없음.** 좌표 `(x, y)` 같은 건 바뀌면 안 되잖아요.
2. **딕셔너리의 키로 쓸 수 있음.** (리스트는 키로 못 씀)
3. **약간 더 빠름.**

### 괄호는 사실 생략해도 됩니다

```python
point = 3, 5       # 소괄호 없이도 튜플
print(point)        # (3, 5)
print(type(point))  # <class 'tuple'>
```

### 함수가 여러 값을 돌려줄 때

함수에서 여러 값을 돌려주면 자동으로 튜플이 돼요. 이건 진짜 자주 씁니다.

```python
def get_min_max(numbers):
    return min(numbers), max(numbers)

result = get_min_max([3, 7, 1, 9, 4])
print(result)         # (1, 9)
print(type(result))   # <class 'tuple'>

# 풀어서 받기 (unpacking)
mn, mx = get_min_max([3, 7, 1, 9, 4])
print(mn, mx)   # 1 9
```

이 마지막 패턴(`mn, mx = ...`)을 **언패킹(unpacking)** 이라고 부르는데, 머신러닝에서 **`X_train, X_test, y_train, y_test = train_test_split(...)`** 처럼 정말 정말 자주 봅니다. 외워두세요.

---

## 3. 셋 (set)

**중복을 자동으로 없애주고, 순서가 없는** 자료구조예요.

```python
fruits = {"사과", "바나나", "딸기"}    # 중괄호 { }

print(fruits)
# {'딸기', '사과', '바나나'}   ← 순서가 들어간 순서가 아닐 수 있음
```

### 중복 제거가 자동

리스트에서 중복을 빠르게 없애고 싶을 때 정말 편합니다.

```python
numbers = [1, 2, 2, 3, 3, 3, 4, 5]
unique = set(numbers)
print(unique)    # {1, 2, 3, 4, 5}

# 다시 리스트로
unique_list = list(unique)
print(unique_list)
```

### 메소드들

```python
fruits = {"사과", "바나나"}

fruits.add("딸기")
print(fruits)

fruits.update(["포도", "수박"])
print(fruits)

fruits.remove("바나나")
print(fruits)
```

### 집합 연산

수학에서 배웠던 집합 연산도 됩니다.

```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print(A | B)    # {1, 2, 3, 4, 5, 6}   합집합
print(A & B)    # {3, 4}                교집합
print(A - B)    # {1, 2}                차집합
```

### 셋의 단점

순서가 없으니까 **인덱스로 접근할 수 없어요.**

```python
fruits = {"사과", "바나나", "딸기"}
fruits[0]    # ❌ TypeError !
```

순서가 중요하면 리스트를, 중복 제거가 중요하면 셋을 쓰세요.

---

## 4. 딕셔너리 (dict)

가장 강력하고 가장 많이 쓰는 자료구조 중 하나예요. **키-값 쌍**으로 저장합니다.

### 만들기

중괄호 `{ }` 안에 `키: 값` 쌍을 콤마로 구분해서 넣어요.

```python
person = {
    "name": "철수",
    "age": 25,
    "height": 175.5,
}

print(person)
# {'name': '철수', 'age': 25, 'height': 175.5}
```

### 값 꺼내기 (키로)

리스트는 인덱스(0, 1, 2)로 꺼냈죠. 딕셔너리는 **키**로 꺼냅니다.

```python
print(person["name"])    # 철수
print(person["age"])     # 25
```

### 값 추가 / 수정

```python
person["weight"] = 70    # 새 키 추가
print(person)

person["age"] = 26       # 기존 키 수정
print(person)
```

### 키 존재 확인

```python
print("name" in person)        # True
print("address" in person)     # False
```

### 안전하게 꺼내기: .get()

존재하지 않는 키로 접근하면 에러가 나요.

```python
print(person["address"])    # KeyError !
```

`.get()`을 쓰면 에러 대신 `None`을 돌려줘요.

```python
print(person.get("address"))         # None
print(person.get("address", "정보없음"))   # 정보없음 (기본값 지정)
```

### 키 / 값 / 쌍 모두 가져오기

```python
person = {"name": "철수", "age": 25}

print(person.keys())     # dict_keys(['name', 'age'])
print(person.values())   # dict_values(['철수', 25])
print(person.items())    # dict_items([('name', '철수'), ('age', 25)])
```

다음 글의 for문에서 정말 자주 씁니다.

```python
for key, value in person.items():
    print(f"{key} → {value}")
```

### 딕셔너리 안에 딕셔너리

복잡한 데이터를 표현할 때 유용해요.

```python
students = {
    "철수": {"age": 25, "score": 85},
    "영희": {"age": 24, "score": 92},
    "민수": {"age": 26, "score": 78},
}

print(students["영희"]["score"])    # 92
```

JSON 데이터(웹에서 받는 데이터)가 보통 이런 모양이에요.

---

## 4가지 그릇 비교표

| | 리스트 | 튜플 | 셋 | 딕셔너리 |
|---|---|---|---|---|
| **만들 때** | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` | `{"a": 1}` |
| **순서** | O | O | X | O (3.7+) |
| **중복** | O | O | X | 키 X, 값 O |
| **수정** | O | X | O | O |
| **인덱스 접근** | `lst[0]` | `tpl[0]` | X | `dct["key"]` |

---

## 자료구조 변환

다른 자료구조로 변환할 수 있어요.

```python
# 리스트 ↔ 튜플
list((1, 2, 3))      # [1, 2, 3]
tuple([1, 2, 3])     # (1, 2, 3)

# 중복 없애기
list(set([1, 2, 2, 3]))    # [1, 2, 3]

# 딕셔너리에서 키만 / 값만 리스트로
list(person.keys())     # ['name', 'age']
list(person.values())   # ['철수', 25]
```

---

## 정리: 한 페이지 요약

```python
# 리스트: 가장 많이 씀
fruits = ["사과", "바나나"]
fruits[0]                # 인덱싱
fruits[1:3]              # 슬라이싱
fruits.append("딸기")     # 추가
len(fruits)              # 길이

# 튜플: 못 바꿈, 함수 반환에 자주 씀
point = (3, 5)
x, y = point             # 언패킹

# 셋: 중복 제거에 좋음
unique = set([1, 2, 2, 3])    # {1, 2, 3}

# 딕셔너리: 이름표 붙은 데이터
person = {"name": "철수", "age": 25}
person["name"]           # 값 접근
person.get("address")    # 안전한 접근
person["height"] = 170   # 추가/수정
```

---

## 자주 묻는 질문

> **Q. 리스트만 잘 써도 되는 것 같은데, 왜 4가지나 있어요?**
>
> 각자 잘하는 게 있어요.
> - 데이터를 자주 바꾼다 → 리스트
> - 한 번 정하면 안 바꾼다 → 튜플
> - 중복을 빠르게 검사한다 → 셋
> - 이름으로 찾고 싶다 → 딕셔너리
>
> 그리고 머신러닝에서 데이터의 모양에 따라 적합한 자료구조가 달라요. 익숙해질수록 본능적으로 고르게 됩니다.

> **Q. 리스트 인덱스가 0부터 시작하는 게 너무 헷갈려요.**
>
> 거의 모든 프로그래밍 언어가 그래요. 처음에는 헷갈리지만 한 달만 쓰시면 자연스러워집니다. 0부터 시작하는 이유는 메모리 주소 계산이 편해서인데, 깊이 알 필요는 없어요.

> **Q. `dict`의 `for k, v in person.items()` 이게 뭐예요?**
>
> 다음 글(제어문)에서 자세히 다룹니다. 미리 보면, 딕셔너리의 키-값 쌍을 하나씩 꺼내서 `k`, `v`에 담는 거예요. 자주 씁니다.

➡️ 다음: [1.5 제어문: 조건과 반복](05-제어문.md)
