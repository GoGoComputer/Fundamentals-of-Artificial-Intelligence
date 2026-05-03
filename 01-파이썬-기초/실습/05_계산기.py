"""1장 종합 실습: 계산기 만들기

지금까지 배운 모든 걸 합친 작은 프로그램입니다.
한 줄 한 줄 의미를 떠올리며 실행해 보세요.

Colab에서는 input()이 잘 동작하지 않을 수 있어요.
그럴 땐 본인 컴퓨터에서:
    python 05_계산기.py
이렇게 실행하시거나, Colab에서는 아래의 calculate() 함수만 가져다 쓰셔도 됩니다.
"""


# ============================================================
# 핵심 함수
# ============================================================

def calculate(a, b, op):
    """두 숫자와 연산자를 받아 결과를 돌려준다.

    지원 연산: +, -, *, /, **, %
    잘못된 입력이거나 0으로 나누기는 None을 돌려준다.
    """
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
    elif op == "**":
        return a ** b
    elif op == "%":
        if b == 0:
            return None
        return a % b
    else:
        return None


def parse_number(text):
    """문자열을 숫자로 변환. 실패하면 None."""
    try:
        return float(text)
    except ValueError:
        return None


# ============================================================
# 메인 루프
# ============================================================

def main():
    print("=" * 40)
    print("  파이썬 계산기")
    print("  (q를 입력하면 종료)")
    print("=" * 40)

    history = []   # 계산 기록

    while True:
        print()
        a_text = input("첫 번째 숫자: ").strip()
        if a_text.lower() == "q":
            break

        b_text = input("두 번째 숫자: ").strip()
        if b_text.lower() == "q":
            break

        op = input("연산 (+, -, *, /, **, %): ").strip()
        if op.lower() == "q":
            break

        # 숫자 변환
        a = parse_number(a_text)
        b = parse_number(b_text)

        if a is None or b is None:
            print("→ 숫자가 잘못됐어요. 다시 시도하세요.")
            continue

        # 계산
        result = calculate(a, b, op)
        if result is None:
            print("→ 계산할 수 없습니다 (잘못된 연산이거나 0으로 나누기)")
            continue

        # 결과 출력 + 기록
        record = f"{a} {op} {b} = {result}"
        print(f"→ {record}")
        history.append(record)

    # 종료 시 기록 출력
    if history:
        print("\n" + "=" * 40)
        print("  지금까지의 계산")
        print("=" * 40)
        for i, record in enumerate(history, start=1):
            print(f"  {i}. {record}")

    print("\n계산기를 종료합니다. 안녕히가세요!")


# ============================================================
# 함수 단위 테스트 (실행 전에 검증)
# ============================================================

def test_calculate():
    """calculate 함수가 잘 동작하는지 검증.

    이런 식으로 함수마다 테스트를 적어 두면, 코드를 고칠 때
    뭐가 깨졌는지 바로 알 수 있어요. 이게 좋은 코딩 습관입니다.
    """
    assert calculate(10, 3, "+") == 13
    assert calculate(10, 3, "-") == 7
    assert calculate(10, 3, "*") == 30
    assert calculate(10, 4, "/") == 2.5
    assert calculate(10, 0, "/") is None       # 0으로 나누기
    assert calculate(2, 10, "**") == 1024
    assert calculate(10, 3, "%") == 1
    assert calculate(10, 3, "?") is None       # 잘못된 연산자
    print("[테스트 통과] calculate 함수 OK")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 먼저 함수 테스트
    test_calculate()
    print()

    # 본 프로그램 실행
    main()
