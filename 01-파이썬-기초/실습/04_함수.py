"""1장 6절 실습: 함수

함수가 손에 익으면 여러분은 코드를 '말로 하듯이' 짤 수 있게 됩니다.
실습해 보세요.
"""

# ============================================================
# 1. 가장 단순한 함수
# ============================================================
print("=" * 50)
print("1. 단순한 함수")
print("=" * 50)


def greet():
    print("안녕하세요!")


# 호출
greet()
greet()


# ============================================================
# 2. 매개변수가 있는 함수
# ============================================================
print("\n" + "=" * 50)
print("2. 매개변수")
print("=" * 50)


def greet(name):
    print(f"안녕하세요 {name}님!")


greet("철수")
greet("영희")


# ============================================================
# 3. return으로 값 돌려주기
# ============================================================
print("\n" + "=" * 50)
print("3. return")
print("=" * 50)


def add(a, b):
    return a + b


result = add(3, 5)
print(result)
print(add(10, 20))
print(add(1.5, 2.5))


# ============================================================
# 4. 기본값
# ============================================================
print("\n" + "=" * 50)
print("4. 기본값")
print("=" * 50)


def greet(name="손님", greeting="안녕하세요"):
    print(f"{greeting} {name}님")


greet()                          # 둘 다 기본값
greet("철수")                     # 첫 번째만
greet("철수", "반가워요")           # 둘 다 지정


# ============================================================
# 5. 키워드 인자
# ============================================================
print("\n" + "=" * 50)
print("5. 키워드 인자")
print("=" * 50)


def order_pizza(size, topping, extra_cheese=False):
    print(f"피자 주문: {size} 사이즈, {topping}")
    if extra_cheese:
        print("  + 치즈 추가")


# 위치 순서대로
order_pizza("L", "페퍼로니")

# 키워드로 (순서 자유)
order_pizza(topping="불고기", size="M")

# 기본값 변경
order_pizza("L", "페퍼로니", extra_cheese=True)


# ============================================================
# 6. 여러 값 돌려주기
# ============================================================
print("\n" + "=" * 50)
print("6. 여러 값 돌려주기")
print("=" * 50)


def get_min_max(numbers):
    return min(numbers), max(numbers)


# 한 번에 두 값을 받음
mn, mx = get_min_max([3, 7, 1, 9, 4])
print(f"min={mn}, max={mx}")

# 튜플로도 받을 수 있음
result = get_min_max([3, 7, 1, 9, 4])
print(result)
print(type(result))


# ============================================================
# 7. 함수가 함수를 부를 수 있어요
# ============================================================
print("\n" + "=" * 50)
print("7. 함수의 조합")
print("=" * 50)


def square(x):
    return x * x


def sum_of_squares(a, b):
    return square(a) + square(b)


print(sum_of_squares(3, 4))   # 9 + 16 = 25


# ============================================================
# 8. 실전 예제: 학생 통계 함수들
# ============================================================
print("\n" + "=" * 50)
print("8. 실전: 학생 통계")
print("=" * 50)


def average(scores):
    """리스트의 평균을 돌려준다."""
    if not scores:
        return 0
    return sum(scores) / len(scores)


def is_passing(score, threshold=60):
    """점수가 합격 기준 이상이면 True."""
    return score >= threshold


def count_passing(scores, threshold=60):
    """합격자 수를 센다."""
    count = 0
    for s in scores:
        if is_passing(s, threshold):
            count += 1
    return count


def grade(score):
    """점수를 받아 학점을 돌려준다."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


# 사용
scores = [85, 92, 78, 95, 67, 55, 72, 88]

print(f"평균: {average(scores):.1f}")
print(f"합격자 수 (60점 기준): {count_passing(scores)}명")
print(f"엄격한 기준 (80점): {count_passing(scores, threshold=80)}명")

print("\n[전체 학점]")
for s in scores:
    print(f"  {s}점 → {grade(s)}")


# ============================================================
# 9. docstring (참고)
# ============================================================
print("\n" + "=" * 50)
print("9. docstring 도움말")
print("=" * 50)


def calculate_area(width, height):
    """직사각형의 넓이를 계산한다.

    인자:
        width: 가로 길이
        height: 세로 길이

    반환:
        넓이 (width * height)
    """
    return width * height


# 도움말 보기
help(calculate_area)


print("\n실습 끝! 함수가 손에 익으면 진짜 프로그래밍이 시작됩니다.")
