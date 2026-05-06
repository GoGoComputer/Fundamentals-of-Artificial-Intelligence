"""1장 6절 실습: 함수

함수는 자주 쓰는 코드를 이름 붙은 부품으로 묶어 주는 도구입니다.
이번 실습에서는 정의와 호출, 입력과 반환, 기본값과 조합까지 한 번에 익힙니다.
"""

# 첫 번째 섹션으로 가장 단순한 함수를 소개합니다.
print("=" * 50)
# 지금부터 볼 주제 제목을 출력합니다.
print("1. 단순한 함수")
# 구분선을 다시 출력해 시각적으로 정리합니다.
print("=" * 50)


# greet 함수는 입력 없이 같은 인사말을 출력하는 가장 단순한 예제입니다.
def greet():
    # 함수 본문에서는 실제로 수행할 동작을 들여써서 적습니다.
    print("안녕하세요!")


# 함수를 한 번 호출하면 정의해 둔 인사말이 한 번 실행됩니다.
greet()
# 같은 함수를 다시 호출하면 같은 동작을 여러 번 재사용할 수 있습니다.
greet()


# 이제 입력값을 받는 함수 섹션으로 넘어갑니다.
print("\n" + "=" * 50)
# 매개변수라는 제목을 출력합니다.
print("2. 매개변수")
# 구분선을 출력해 새 섹션을 표시합니다.
print("=" * 50)


# name을 매개변수로 받으면 같은 함수가 여러 사람에게 맞는 인사말을 만들 수 있습니다.
def greet(name):
    # f-string을 이용해 전달받은 이름을 문장 안에 자연스럽게 넣습니다.
    print(f"안녕하세요 {name}님!")


# 철수를 넣어 호출하면 철수에게 맞는 인사말이 출력됩니다.
greet("철수")
# 영희를 넣어 호출하면 같은 함수가 다른 결과를 냅니다.
greet("영희")


# 이번에는 화면에 출력만 하는 함수가 아니라 값을 돌려주는 함수를 봅니다.
print("\n" + "=" * 50)
# return 섹션 제목을 출력합니다.
print("3. return")
# 구분선을 유지합니다.
print("=" * 50)


# add 함수는 두 입력을 받아 더한 결과를 return으로 돌려줍니다.
def add(a, b):
    # return을 만나면 함수는 값을 되돌려주고 바로 종료됩니다.
    return a + b


# add의 결과를 변수에 담으면 나중에 다시 사용할 수 있습니다.
result = add(3, 5)
# 변수에 담긴 결과를 출력해 8이 되었는지 확인합니다.
print(result)
# 함수를 print 안에서 바로 호출해도 같은 결과를 얻을 수 있습니다.
print(add(10, 20))
# 실수끼리도 같은 함수가 잘 동작하는지 확인합니다.
print(add(1.5, 2.5))


# 기본값 섹션에서는 인자를 생략해도 동작하는 함수를 살펴봅니다.
print("\n" + "=" * 50)
# 섹션 제목을 출력합니다.
print("4. 기본값")
# 구분선을 출력합니다.
print("=" * 50)


# 기본값을 넣어 두면 호출할 때 아무것도 넘기지 않아도 함수가 동작합니다.
def greet(name="손님", greeting="안녕하세요"):
    # 전달된 값이 있으면 그 값을 쓰고 없으면 기본값을 사용합니다.
    print(f"{greeting} {name}님")


# 아무 인자 없이 호출하면 두 기본값이 그대로 사용됩니다.
greet()
# 첫 번째 인자만 넘기면 name만 바뀌고 greeting은 기본값을 유지합니다.
greet("철수")
# 두 값을 모두 넘기면 기본값 대신 새 값들이 사용됩니다.
greet("철수", "반가워요")


# 이제 매개변수 이름을 직접 적는 키워드 인자를 실습합니다.
print("\n" + "=" * 50)
# 섹션 제목을 출력합니다.
print("5. 키워드 인자")
# 구분선을 출력합니다.
print("=" * 50)


# 피자 주문 함수는 값이 여러 개인 함수를 읽는 연습에 잘 맞는 예제입니다.
def order_pizza(size, topping, extra_cheese=False):
    # 기본 주문 정보는 항상 먼저 출력합니다.
    print(f"피자 주문: {size} 사이즈, {topping}")
    # extra_cheese가 True일 때만 추가 메시지를 출력합니다.
    if extra_cheese:
        print("  + 치즈 추가")


# 위치 인자만 사용하면 정의된 순서대로 값을 넘기면 됩니다.
order_pizza("L", "페퍼로니")

# 키워드 인자를 쓰면 순서를 바꿔도 의미가 분명하게 유지됩니다.
order_pizza(topping="불고기", size="M")

# 기본값이 있는 인자도 필요할 때는 명시적으로 바꿀 수 있습니다.
order_pizza("L", "페퍼로니", extra_cheese=True)


# 이번에는 함수가 값을 하나보다 많이 돌려주는 경우를 다룹니다.
print("\n" + "=" * 50)
# 제목을 출력합니다.
print("6. 여러 값 돌려주기")
# 구분선을 출력합니다.
print("=" * 50)


# 최소값과 최대값을 같이 돌려주는 함수는 다중 반환의 전형적인 예제입니다.
def get_min_max(numbers):
    # 콤마로 값을 나란히 돌려주면 실제로는 튜플 형태가 됩니다.
    return min(numbers), max(numbers)


# 언패킹으로 두 값을 바로 나눠 받으면 코드가 깔끔해집니다.
mn, mx = get_min_max([3, 7, 1, 9, 4])
# 각각의 값이 잘 담겼는지 출력해 확인합니다.
print(f"min={mn}, max={mx}")

# 같은 결과를 하나의 변수로 받으면 튜플이라는 점도 확인할 수 있습니다.
result = get_min_max([3, 7, 1, 9, 4])
# 튜플 모양이 그대로 출력됩니다.
print(result)
# 실제 자료형도 tuple인지 확인합니다.
print(type(result))


# 함수 안에서 다른 함수를 호출하는 조합 예제로 넘어갑니다.
print("\n" + "=" * 50)
# 섹션 제목을 출력합니다.
print("7. 함수의 조합")
# 구분선을 출력합니다.
print("=" * 50)


# square 함수는 숫자 하나를 받아 제곱을 돌려줍니다.
def square(x):
    # x에 x를 곱해 제곱값을 계산합니다.
    return x * x


# sum_of_squares 함수는 square 함수를 재사용해 더 큰 기능을 만듭니다.
def sum_of_squares(a, b):
    # 각 값을 제곱한 뒤 더한 결과를 돌려줍니다.
    return square(a) + square(b)


# 3과 4를 넣으면 9와 16의 합인 25가 나오는지 확인합니다.
print(sum_of_squares(3, 4))


# 조금 더 현실적인 학생 통계 예제로 함수들을 여러 개 나눠 봅니다.
print("\n" + "=" * 50)
# 섹션 제목을 출력합니다.
print("8. 실전: 학생 통계")
# 구분선을 출력합니다.
print("=" * 50)


# average 함수는 점수 리스트의 평균을 계산합니다.
def average(scores):
    """리스트의 평균을 돌려준다."""
    # 빈 리스트가 들어오면 0으로 처리해 나누기 에러를 막습니다.
    if not scores:
        return 0
    # 비어 있지 않다면 합계를 길이로 나눠 평균을 계산합니다.
    return sum(scores) / len(scores)


# is_passing 함수는 한 점수가 합격 기준을 넘는지 판정합니다.
def is_passing(score, threshold=60):
    """점수가 합격 기준 이상이면 True."""
    # 비교 결과 자체가 True 또는 False이므로 그대로 돌려주면 됩니다.
    return score >= threshold


# count_passing 함수는 여러 점수 중 합격자가 몇 명인지 세어 줍니다.
def count_passing(scores, threshold=60):
    """합격자 수를 센다."""
    # 합격자 수를 담을 카운터를 0으로 시작합니다.
    count = 0
    # 점수를 하나씩 보면서 합격 여부를 확인합니다.
    for s in scores:
        # 합격이면 카운터를 1 증가시킵니다.
        if is_passing(s, threshold):
            count += 1
    # 최종 합격자 수를 돌려줍니다.
    return count


# grade 함수는 점수를 받아 문자 학점으로 바꿔 줍니다.
def grade(score):
    """점수를 받아 학점을 돌려준다."""
    # 가장 높은 점수대부터 차례대로 검사합니다.
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    # 위 조건에 모두 걸리지 않으면 F를 돌려줍니다.
    return "F"


# 함수들을 시험해 볼 점수 목록을 하나 준비합니다.
scores = [85, 92, 78, 95, 67, 55, 72, 88]

# average 함수 결과를 소수점 한 자리까지 출력합니다.
print(f"평균: {average(scores):.1f}")
# 기본 합격 기준인 60점으로 합격자 수를 계산합니다.
print(f"합격자 수 (60점 기준): {count_passing(scores)}명")
# threshold를 바꾸면 더 엄격한 기준으로도 같은 함수를 재사용할 수 있습니다.
print(f"엄격한 기준 (80점): {count_passing(scores, threshold=80)}명")

# 각 점수를 문자 학점으로 변환해 보는 반복문도 실행합니다.
print("\n[전체 학점]")
# 점수 하나마다 grade 함수를 호출해 결과를 출력합니다.
for s in scores:
    print(f"  {s}점 → {grade(s)}")


# 마지막으로 함수 설명을 함수 안에 적는 docstring을 간단히 봅니다.
print("\n" + "=" * 50)
# 섹션 제목을 출력합니다.
print("9. docstring 도움말")
# 구분선을 출력합니다.
print("=" * 50)


# calculate_area 함수는 docstring이 실제로 어떤 모양인지 보여 주는 예제입니다.
def calculate_area(width, height):
    """직사각형의 넓이를 계산한다.

    인자:
        width: 가로 길이
        height: 세로 길이

    반환:
        넓이 (width * height)
    """
    # 가로와 세로를 곱해 넓이를 계산해 돌려줍니다.
    return width * height


# help를 호출하면 방금 적은 docstring이 도움말로 보입니다.
help(calculate_area)


# 마무리 문장을 출력해 함수 실습을 마칩니다.
print("\n실습 끝! 함수가 손에 익으면 진짜 프로그래밍이 시작됩니다.")
