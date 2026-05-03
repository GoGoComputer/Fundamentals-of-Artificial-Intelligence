"""1장 4절 실습: 리스트, 튜플, 셋, 딕셔너리

각 섹션을 차례로 실행하시면서 결과를 확인하세요.
중간에 본인이 변형해서 실험해 보시는 게 가장 좋아요.
"""

# ============================================================
# 1. 리스트 (가장 자주 씀)
# ============================================================
print("=" * 50)
print("1. 리스트")
print("=" * 50)

fruits = ["사과", "바나나", "딸기"]
print(fruits)

# 인덱싱
print("첫 번째:", fruits[0])
print("두 번째:", fruits[1])
print("마지막:", fruits[-1])

# 슬라이싱
nums = [10, 20, 30, 40, 50]
print(nums[1:4])     # [20, 30, 40]
print(nums[:3])      # [10, 20, 30]
print(nums[2:])      # [30, 40, 50]
print(nums[::-1])    # [50, 40, 30, 20, 10]   역순!

# 수정
fruits[1] = "포도"
print("수정 후:", fruits)

# 추가/삭제
fruits.append("수박")
print("추가 후:", fruits)

fruits.remove("사과")
print("삭제 후:", fruits)

# 정렬
nums = [3, 1, 4, 1, 5, 9, 2, 6]
nums.sort()
print("정렬:", nums)

nums.sort(reverse=True)
print("역순 정렬:", nums)

# 길이, 합, 최대, 최소
print("개수:", len(nums))
print("합:", sum(nums))
print("최대:", max(nums))
print("최소:", min(nums))

# 2차원 리스트
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
print("2x2 위치:", matrix[1][1])    # 5


# ============================================================
# 2. 튜플 (못 바꾸는 리스트)
# ============================================================
print("\n" + "=" * 50)
print("2. 튜플")
print("=" * 50)

point = (3, 5)
colors = ("빨강", "초록", "파랑")
print(point)
print(colors[1])

# 언패킹 (정말 자주 씀!)
x, y = point
print(f"x={x}, y={y}")

# 함수가 여러 값을 돌려줄 때
def get_min_max(numbers):
    return min(numbers), max(numbers)

mn, mx = get_min_max([3, 7, 1, 9, 4])
print(f"min={mn}, max={mx}")


# ============================================================
# 3. 셋 (중복 없음, 순서 없음)
# ============================================================
print("\n" + "=" * 50)
print("3. 셋")
print("=" * 50)

# 중복 자동 제거
nums = [1, 2, 2, 3, 3, 3, 4, 5]
unique = set(nums)
print(unique)

# 집합 연산
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print("합집합:", A | B)
print("교집합:", A & B)
print("차집합:", A - B)

# 추가/삭제
fruits = {"사과", "바나나"}
fruits.add("딸기")
print(fruits)


# ============================================================
# 4. 딕셔너리 (이름표 붙은 데이터)
# ============================================================
print("\n" + "=" * 50)
print("4. 딕셔너리")
print("=" * 50)

person = {
    "name": "철수",
    "age": 25,
    "height": 175.5,
}

# 값 접근
print(person["name"])
print(person.get("address", "정보없음"))   # 안전한 접근

# 추가/수정
person["weight"] = 70
person["age"] = 26
print(person)

# 키, 값, 쌍 가져오기
print(list(person.keys()))
print(list(person.values()))

# 키 존재 확인
print("name 있나?", "name" in person)

# 중첩
students = {
    "철수": {"age": 25, "score": 85},
    "영희": {"age": 24, "score": 92},
}
print("영희 점수:", students["영희"]["score"])


print("\n실습 끝! 4가지 그릇을 자유자재로 다루실 수 있게 되시면 1장 절반 끝입니다.")
