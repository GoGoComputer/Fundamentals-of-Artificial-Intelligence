"""1장 5절 실습: if, for, while

제어문은 머신러닝 코드에서 매번 만나게 되는 도구입니다.
손에 익을 때까지 직접 변형해 보세요.
"""

# ============================================================
# 1. 조건문 (if / elif / else)
# ============================================================
print("=" * 50)
print("1. 조건문")
print("=" * 50)

# 단순한 if
age = 25
if age >= 19:
    print("성인입니다")

# if-else
score = 75
if score >= 60:
    print("합격")
else:
    print("불합격")

# if-elif-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(f"학점: {grade}")

# 빈 컨테이너는 False
items = []
if not items:
    print("리스트가 비어있어요")

text = ""
if not text:
    print("문자열이 비어있어요")


# ============================================================
# 2. for문 (가장 자주 씀)
# ============================================================
print("\n" + "=" * 50)
print("2. for문")
print("=" * 50)

# 리스트 순회
fruits = ["사과", "바나나", "딸기"]
for fruit in fruits:
    print(f"  - {fruit}")

# range
print("\n0부터 4까지:")
for i in range(5):
    print(i, end=" ")
print()

print("\n2부터 7 미만, 1씩:")
for i in range(2, 7):
    print(i, end=" ")
print()

print("\n0부터 10 미만, 2씩:")
for i in range(0, 10, 2):
    print(i, end=" ")
print()

# enumerate (인덱스도 같이)
print("\nenumerate:")
for i, fruit in enumerate(fruits):
    print(f"  {i}번째: {fruit}")

# 딕셔너리 순회
print("\n딕셔너리 순회:")
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
}
for key, value in config.items():
    print(f"  {key}: {value}")

# 누적 합계 패턴
nums = [10, 20, 30, 40, 50]
total = 0
for n in nums:
    total += n
print(f"\n누적 합: {total}")

# (있는 거 쓰는 법)
print(f"sum() 한 번에: {sum(nums)}")


# ============================================================
# 3. while문
# ============================================================
print("\n" + "=" * 50)
print("3. while문")
print("=" * 50)

# 0부터 4까지
count = 0
while count < 5:
    print(count, end=" ")
    count += 1
print()

# 조건이 만족될 때까지
n = 100
steps = 0
while n > 1:
    if n % 2 == 0:
        n = n // 2
    else:
        n = 3 * n + 1
    steps += 1
print(f"100부터 시작해서 1이 되기까지 {steps}단계")


# ============================================================
# 4. break와 continue
# ============================================================
print("\n" + "=" * 50)
print("4. break와 continue")
print("=" * 50)

# break: 즉시 종료
print("5에서 멈춤:")
for n in range(20):
    if n == 5:
        break
    print(n, end=" ")
print()

# continue: 한 번 건너뜀
print("\n홀수만:")
for n in range(10):
    if n % 2 == 0:
        continue
    print(n, end=" ")
print()


# ============================================================
# 5. 종합: 학생 점수 처리
# ============================================================
print("\n" + "=" * 50)
print("5. 종합 예제: 학생 점수")
print("=" * 50)

students = {
    "철수": 85,
    "영희": 92,
    "민수": 78,
    "수진": 95,
    "지호": 67,
}

print("[전체 학생]")
for name, score in students.items():
    print(f"  {name}: {score}점")

# 평균
avg = sum(students.values()) / len(students)
print(f"\n평균: {avg:.1f}점")

# 합격자
print("\n[합격자 (80점 이상)]")
for name, score in students.items():
    if score >= 80:
        print(f"  {name}: {score}점")

# 최고점
best_name, best_score = "", 0
for name, score in students.items():
    if score > best_score:
        best_name, best_score = name, score
print(f"\n최고점: {best_name} ({best_score}점)")


# ============================================================
# 보너스: 리스트 컴프리헨션
# ============================================================
print("\n" + "=" * 50)
print("보너스: 리스트 컴프리헨션")
print("=" * 50)

# for문 버전
evens = []
for n in range(10):
    if n % 2 == 0:
        evens.append(n)
print("for문:", evens)

# 한 줄 버전 (같은 결과)
evens = [n for n in range(10) if n % 2 == 0]
print("컴프리헨션:", evens)

# 제곱 리스트
squares = [n * n for n in range(10)]
print("제곱:", squares)


print("\n실습 끝! 제어문은 머신러닝의 모든 곳에서 만납니다.")
