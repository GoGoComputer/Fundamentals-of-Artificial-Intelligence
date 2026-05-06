"""PDF 내보내기 스크립트

book/print.html → PDF

사용법:
    python scripts/export_pdf.py                  # 기본 (book/AI-기초.pdf)
    python scripts/export_pdf.py -o output.pdf    # 출력 경로 지정

필요 조건:
    macOS: /Applications/Google Chrome.app  (또는 Chromium, Brave)
    Linux: google-chrome 또는 chromium-browser
    Windows: chrome.exe (PATH에 있어야 함)

Chrome 없이 WeasyPrint 사용:
    pip install weasyprint
    python scripts/export_pdf.py --weasyprint
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRINT_HTML = ROOT / "book" / "print.html"
DEFAULT_OUTPUT = ROOT / "book" / "AI-기초.pdf"

# macOS/Linux/Windows 순서로 Chrome 후보 경로를 탐색합니다.
CHROME_CANDIDATES = [
    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    # Linux
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    # Windows (PATH 기준)
    "chrome",
]


def find_chrome() -> str | None:
    """사용 가능한 Chrome/Chromium 실행 파일 경로를 반환합니다."""
    for candidate in CHROME_CANDIDATES:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        # PATH 탐색
        result = subprocess.run(
            ["which", candidate],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def export_with_chrome(output: Path) -> bool:
    """Chrome headless로 PDF를 생성합니다."""
    chrome = find_chrome()
    if not chrome:
        return False

    print(f"Chrome 발견: {chrome}")
    print(f"PDF 생성 중... ({PRINT_HTML} → {output})")

    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        "--disable-dev-shm-usage",
        f"--print-to-pdf={output}",
        "--print-to-pdf-no-header",
        str(PRINT_HTML),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and output.exists():
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"✅ 완료: {output}  ({size_mb:.1f} MB)")
        return True
    else:
        print(f"Chrome 실패: {result.stderr.strip()}", file=sys.stderr)
        return False


def export_with_weasyprint(output: Path) -> bool:
    """WeasyPrint로 PDF를 생성합니다."""
    try:
        from weasyprint import HTML
    except ImportError:
        print("WeasyPrint가 설치되지 않았습니다.", file=sys.stderr)
        print("설치: pip install weasyprint", file=sys.stderr)
        return False

    print(f"WeasyPrint로 PDF 생성 중... ({PRINT_HTML} → {output})")

    try:
        HTML(filename=str(PRINT_HTML)).write_pdf(str(output))
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"✅ 완료: {output}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"WeasyPrint 실패: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="mdBook → PDF 내보내기")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 PDF 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--weasyprint",
        action="store_true",
        help="Chrome 대신 WeasyPrint 사용",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="PDF 생성 전에 책을 먼저 빌드 (sync + mdbook build)",
    )
    args = parser.parse_args()

    # 1. 먼저 책 빌드 (--build 옵션)
    if args.build:
        print("책 빌드 중...")
        sync = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sync_book.py")],
            cwd=ROOT,
        )
        if sync.returncode != 0:
            sys.exit("sync_book.py 실패")
        build = subprocess.run(["mdbook", "build"], cwd=ROOT)
        if build.returncode != 0:
            sys.exit("mdbook build 실패")

    # 2. print.html 존재 확인
    if not PRINT_HTML.exists():
        print(f"print.html이 없습니다: {PRINT_HTML}", file=sys.stderr)
        print("먼저 책을 빌드하세요:", file=sys.stderr)
        print("  python scripts/build_book.py", file=sys.stderr)
        print("  또는: python scripts/export_pdf.py --build", file=sys.stderr)
        sys.exit(1)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 3. PDF 생성
    if args.weasyprint:
        ok = export_with_weasyprint(output)
    else:
        ok = export_with_chrome(output)
        if not ok:
            print("\nChrome을 찾을 수 없습니다. WeasyPrint로 재시도합니다...", file=sys.stderr)
            ok = export_with_weasyprint(output)

    if not ok:
        print("\n--- PDF 생성 실패 ---", file=sys.stderr)
        print("다음 중 하나를 설치하세요:", file=sys.stderr)
        print("  1) Google Chrome / Chromium / Brave", file=sys.stderr)
        print("  2) pip install weasyprint", file=sys.stderr)
        print("\n또는 브라우저에서 직접 인쇄하세요:", file=sys.stderr)
        print(f"  open {PRINT_HTML}  →  파일 > 인쇄 > PDF로 저장", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
