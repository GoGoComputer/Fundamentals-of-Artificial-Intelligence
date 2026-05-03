"""1장 1~3절 실습: print, 변수, 자료형, 연산자

Colab이나 본인 컴퓨터에서 한 줄씩 실행해 보세요.
한꺼번에 다 돌리시면 안 됩니다. 한 줄씩, 결과를 보면서요.
"""

# ============================================================
# 1. print 와 변수
# ============================================================
print("=" * 50)
print("1. print와 변수")
print("=" * 50)

# 가장 단순한 출력
print("Hello, World!")

# 여러 개 출력
print("이름:", "철수", "나이:", 25)

# 변수에 담기
name = "철수"
age = 25
height = 175.5

print(name, age, height)

# 변수 다시 할당
name = "영희"
print("이제 name은:", name)


# ============================================================
# 2. 자료형 (int, float, str, bool, None)
# ============================================================
print("\n" + "=" * 50)
print("2. 자료형")
print("=" * 50)

# 다섯 가지 자료형
a = 25          # int
b = 3.14        # float
c = "hello"    # str
d = True        # bool
e = None        # NoneType

print(type(a))
print(type(b))
print(type(c))
print(type(d))
print(type(e))

# 자료형 변환
print(str(25))         # "25"
print(int("100"))     # 100
print(float(5))        # 5.0
print(int(3.7))        # 3 (소수점 버림)
print(round(3.7))      # 4 (반올림)

# 문자열 다루기
greeting = "안녕"
name = "철수"

print(greeting + " " + name)        # 이어 붙이기
print("야호! " * 3)                  # 반복
print(len(greeting))                 # 길이

# f-string (정말 자주 씀)
print(f"제 이름은 {name}이고 나이는 {age}살입니다.")
print(f"내년 나이: {age + 1}")
print(f"파이: {3.14159265:.2f}")     # 소수점 2자리


# ============================================================
# 3. 연산자
# ============================================================
print("\n" + "=" * 50)
print("3. 연산자")
print("=" * 50)

# 산술 연산자
print(10 + 3)     # 13
print(10 - 3)     # 7
print(10 * 3)     # 30
print(10 / 3)     # 3.333... (실수)
print(10 // 3)    # 3 (몫)
print(10 % 3)     # 1 (나머지)
print(2 ** 10)    # 1024 (거듭제곱)
print(16 ** 0.5)  # 4.0 (제곱근)

# 비교 연산자 (결과는 True/False)
print(5 == 5)     # True
print(5 != 3)     # True
print(5 > 3)      # True
print(5 < 3)      # False

# 논리 연산자
age = 25
has_license = True

can_drive = (age >= 18) and has_license
print("운전 가능:", can_drive)

is_student = False
is_senior = True
can_discount = is_student or is_senior
print("할인 가능:", can_discount)

# 할당 단축
count = 0
count += 1   # count = count + 1
count += 1
count += 1
print("count:", count)   # 3

# 멤버십
fruits = ["사과", "바나나", "딸기"]
print("사과 있나?", "사과" in fruits)
print("포도 있나?", "포도" in fruits)


print("\n실습 끝! 결과를 한 줄씩 비교해 보세요.")
