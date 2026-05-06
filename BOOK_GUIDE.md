# 📖 인공지능 기초 — 한 권의 책

이 프로젝트가 이제 **한 권의 완성된 책**으로 변환되었습니다!

## 📚 책 보기

### 방법 1: 로컬 웹 서버로 보기 (가장 권장)

```bash
cd /Users/mo/DEV/devStudy/Fundamentals-of-Artificial-Intelligence
mdbook serve
```

그러면 `http://localhost:3000` 에서 책을 볼 수 있습니다.

**기능:**
- ✅ 실시간 검색 (좌측 검색 창)
- ✅ 목차 탐색 (좌측 메뉴)
- ✅ 다크 모드 / 라이트 모드 전환
- ✅ 텍스트 하이라이트

### 방법 2: HTML 파일 직접 열기

`book/` 폴더 내 HTML 파일들을 브라우저에서 직접 열 수 있습니다:

- **`book/index.html`** — 목차 페이지
- **`book/intro.html`** — 서문
- **`book/00-intro/01-why-ai.html`** — 각 장 페이지

### 방법 3: PDF 생성

```bash
# Chrome / Chromium / Brave가 설치되어 있으면 바로 실행
python scripts/export_pdf.py

# 출력 경로 지정
python scripts/export_pdf.py -o my-book.pdf

# 책 빌드 + PDF 한 번에
python scripts/export_pdf.py --build

# Chrome이 없을 때 WeasyPrint 사용
pip install weasyprint
python scripts/export_pdf.py --weasyprint
```

> Chrome이 없고 WeasyPrint도 없다면, 브라우저에서 `book/print.html`을 열고  
> **파일 → 인쇄 → PDF로 저장** 으로 직접 내보낼 수 있습니다.

## 📁 파일 구조

```
Fundamentals-of-Artificial-Intelligence/
├── book.toml           ← 책 설정
├── src/                ← 원본 마크다운 파일들
│   ├── intro.md        ← 서문
│   ├── SUMMARY.md      ← 목차 (자동으로 생성됨)
│   ├── 00-intro/       ← 0장
│   ├── 01-python-basics/
│   ├── 02-ml-classification/
│   ├── 03-ml-regression/
│   ├── 04-dl-classification/
│   ├── 05-dl-regression/
│   ├── 06-production/
│   └── appendix/       ← 부록 (수학, 용어 사전 등)
├── book/               ← 빌드된 HTML 책 (웹서버로 열기)
└── README.md
```

## 🎯 책의 구성

| 장 | 제목 | 페이지 |
|----|------|--------|
| 0 | 시작하기 전에 | 4 |
| 1 | 파이썬 기초 | 28 |
| 2 | 머신러닝 — 분류 | 65 |
| 3 | 머신러닝 — 회귀 | 42 |
| 4 | 딥러닝 — 분류 | 71 |
| 5 | 딥러닝 — 회귀 | 58 |
| 6 | 프로덕션으로 가는 길 | 36 |
| 부록 | 수학, 용어사전, FAQ 등 | 80 |

**총 약 250~300페이지 분량**

## 🔧 책 수정 / 다시 빌드하기

### 1. 강의 원본 수정

이제 책의 원본은 `src/`가 아니라 상위 강의 폴더입니다.

- 이론: `00-시작하기-전에/` ~ `06-프로덕션으로-가는-길/` 아래 원본 `.md`
- 실습: 각 장의 `실습/*.py`
- 책용 `src/`는 위 원본에서 **자동 생성**됩니다.

### 2. 원본을 책 소스로 동기화

```bash
cd /Users/mo/DEV/devStudy/Fundamentals-of-Artificial-Intelligence
python scripts/sync_book.py
```

이 명령은 다음을 자동으로 처리합니다.

- 강의 원본 이론 파일을 `src/`로 반영
- 실습 `.py` 파일을 mdBook용 실습 페이지로 생성
- `src/SUMMARY.md` 자동 재생성
- 책 안 링크를 mdBook 페이지 기준으로 자동 보정

실습 페이지의 코드는 항상 **생략 없는 전체 코드**로 들어갑니다.

### 3. 책 다시 빌드

```bash
cd /Users/mo/DEV/devStudy/Fundamentals-of-Artificial-Intelligence
python scripts/build_book.py
```

이 명령은 `sync_book.py`를 먼저 실행한 뒤 `mdbook build`까지 이어서 수행합니다.

### 4. 변경 사항 실시간 보기

```bash
cd /Users/mo/DEV/devStudy/Fundamentals-of-Artificial-Intelligence
python scripts/watch_book.py
```

이 모드에서는 강의 원본이 바뀌면 자동으로 `src/`를 다시 생성하고, `mdbook serve`로 미리보기를 계속 유지합니다.

### 5. 직접 mdBook만 실행하고 싶을 때

이미 `src/`가 최신 상태라는 것이 확실할 때만 아래 명령을 직접 써도 됩니다.

```bash
mdbook build
mdbook serve
```

## 📖 온라인에 배포하기

### GitHub Pages로 배포

```bash
# 1. build/ 폴더를 git에 추가
git add book/
git commit -m "Build book"

# 2. GitHub에 push
git push origin main

# 3. GitHub 저장소 settings에서 Pages 활성화
# 소스: main branch / docs folder (또는 root)
```

### 기타 호스팅

- **Netlify**: `book/` 폴더 전체 드래그앤드롭
- **Vercel**: GitHub 연동 후 `book/`을 public 폴더로 설정
- **AWS S3**: `book/` 폴더 전체 업로드

## 🎨 책의 모양 커스터마이징

`book.toml` 파일을 수정하여:

```toml
[output.html]
default-theme = "light"    # light, dark, rust, coal, navy, ayu
preferred-dark-theme = "dark"
```

## 🚀 더 해볼 수 있는 것들

- [ ] 책에 색인(index) 추가
- [ ] 각 장에 연습 문제 추가
- [ ] GitHub Actions로 자동 빌드
- [ ] 책 표지 디자인 추가
- [ ] 실습 코드 다운로드 링크 추가

## 📞 피드백

- 오류 발견: GitHub Issues에 등록
- 개선 제안: Pull Request 제출
- 질문: Discussions 탭 사용

---

**지금 바로 `mdbook serve` 를 실행하고 책을 읽어보세요!** 📚✨
