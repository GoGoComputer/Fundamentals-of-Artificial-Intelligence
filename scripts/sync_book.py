from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
SUMMARY_PATH = SRC_ROOT / "SUMMARY.md"


@dataclass(frozen=True)
class ChapterConfig:
    """강의 원본 폴더 하나를 mdBook src/ 폴더 하나로 대응시키는 설정.

    slugs: 원본 파일 이름(확장자 제외) → 책용 영문 파일 이름(확장자 제외).
    원본 폴더에 새 이론 파일을 추가하면 여기에도 한 줄을 추가해야 합니다.
    (예전에는 정렬 순서로 짝을 맞췄는데, 파일 하나만 끼어들어도 내용이 엉뚱한
    페이지로 들어가거나 동기화가 통째로 실패했습니다.)
    """

    summary_heading: str
    source_dir: Path
    theory_dir: Path
    target_dir: Path
    theory_glob: str
    slugs: dict[str, str]
    practice_dir: Path | None = None


CHAPTERS: tuple[ChapterConfig, ...] = (
    ChapterConfig(
        summary_heading="# 시작하기 전에",
        source_dir=ROOT / "00-시작하기-전에",
        theory_dir=ROOT / "00-시작하기-전에",
        target_dir=SRC_ROOT / "00-intro",
        theory_glob="[0-9][0-9]*-*.md",
        slugs={
            "01-왜-지금-AI인가": "01-why-ai",
            "02-AI-ML-DL-차이": "02-ai-ml-dl",
            "03-환경-설정": "03-setup",
            "04-질문하는-법": "04-how-to-ask",
        },
    ),
    ChapterConfig(
        summary_heading="# 1장. 파이썬 기초",
        source_dir=ROOT / "01-파이썬-기초",
        theory_dir=ROOT / "01-파이썬-기초" / "이론",
        target_dir=SRC_ROOT / "01-python-basics",
        theory_glob="*.md",
        slugs={
            "01-첫-만남": "01-hello",
            "01a-파이썬-심화": "01a-python-advanced",
            "02-자료형": "02-data-types",
            "03-연산자": "03-operators",
            "04-자료구조": "04-data-structure",
            "05-제어문": "05-control-flow",
            "06-함수": "06-functions",
            "07-실전-예제": "07-project",
        },
        practice_dir=ROOT / "01-파이썬-기초" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 2장. 머신러닝 — 분류",
        source_dir=ROOT / "02-머신러닝-분류",
        theory_dir=ROOT / "02-머신러닝-분류" / "이론",
        target_dir=SRC_ROOT / "02-ml-classification",
        theory_glob="*.md",
        slugs={
            "01-분류란-무엇인가": "01-what-is-classification",
            "02-데이터-만나기": "02-meet-data",
            "03-데이터-준비": "03-data-preparation",
            "04-첫-모델-SVM": "04-svm-intro",
            "04a-SVM-깊게": "04a-svm-deep",
            "05-평가-지표": "05-evaluation-metrics",
            "05a-손실-함수-깊게": "05a-loss-function",
            "06-여러-모델": "06-multiple-models",
            "07-하이퍼파라미터": "07-hyperparameter-tuning",
            "08-현업-체크리스트": "08-production-checklist",
            "08a-클래스-불균형": "08a-class-imbalance",
        },
        practice_dir=ROOT / "02-머신러닝-분류" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 3장. 머신러닝 — 회귀",
        source_dir=ROOT / "03-머신러닝-회귀",
        theory_dir=ROOT / "03-머신러닝-회귀" / "이론",
        target_dir=SRC_ROOT / "03-ml-regression",
        theory_glob="*.md",
        slugs={
            "01-회귀란-무엇인가": "01-what-is-regression",
            "02-데이터-만나기": "02-meet-data",
            "03-선형회귀": "03-linear-regression",
            "04-평가-지표": "04-evaluation-metrics",
            "05-정규화": "05-regularization",
            "06-트리-회귀": "06-tree-regression",
            "07-현업-체크리스트": "07-production-checklist",
        },
        practice_dir=ROOT / "03-머신러닝-회귀" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 4장. 딥러닝 — 분류",
        source_dir=ROOT / "04-딥러닝-분류",
        theory_dir=ROOT / "04-딥러닝-분류" / "이론",
        target_dir=SRC_ROOT / "04-dl-classification",
        theory_glob="*.md",
        slugs={
            "01-신경망이란": "01-neural-networks",
            "02-PyTorch-시작": "02-pytorch-intro",
            "03-첫-신경망": "03-first-neural-network",
            "04-학습-과정": "04-training-process",
            "04a-경사하강법-깊게": "04a-gradient-descent",
            "05-깊게": "05-deep-dive",
            "05a-Batch-Normalization-깊게": "05a-batch-norm",
            "06-GPU-활용": "06-gpu",
            "07-현업-체크리스트": "07-production-checklist",
        },
        practice_dir=ROOT / "04-딥러닝-분류" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 5장. 딥러닝 — 회귀",
        source_dir=ROOT / "05-딥러닝-회귀",
        theory_dir=ROOT / "05-딥러닝-회귀" / "이론",
        target_dir=SRC_ROOT / "05-dl-regression",
        theory_glob="*.md",
        slugs={
            "01-회귀-신경망": "01-neural-regression",
            "02-과적합이란": "02-overfitting",
            "02a-데이터-증강": "02a-data-augmentation",
            "03-Early-Stopping": "03-early-stopping",
            "04-Dropout": "04-dropout",
            "05-Weight-Decay": "05-weight-decay",
            "06-종합": "06-combining-techniques",
            "07-현업-체크리스트": "07-production-checklist",
        },
        practice_dir=ROOT / "05-딥러닝-회귀" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 6장. 프로덕션으로 가는 길",
        source_dir=ROOT / "06-프로덕션으로-가는-길",
        theory_dir=ROOT / "06-프로덕션으로-가는-길" / "이론",
        target_dir=SRC_ROOT / "06-production",
        theory_glob="*.md",
        slugs={
            "01-모델은-서비스가-아니다": "01-model-vs-service",
            "02-패키징-저장": "02-model-packaging",
            "03-FastAPI-서빙": "03-fastapi-serving",
            "04-Docker": "04-docker",
            "05-모니터링": "05-monitoring",
            "06-MLOps": "06-mlops",
        },
        practice_dir=ROOT / "06-프로덕션으로-가는-길" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 7장. 더 넓은 AI 세계",
        source_dir=ROOT / "07-더-넓은-AI-세계",
        theory_dir=ROOT / "07-더-넓은-AI-세계" / "이론",
        target_dir=SRC_ROOT / "07-wider-ai",
        theory_glob="*.md",
        slugs={
            "01-비지도학습": "01-unsupervised-learning",
            "02-CNN-입문": "02-cnn-intro",
            "03-RNN-LSTM-GRU": "03-rnn-lstm-gru",
            "04-Transformer-Attention": "04-transformer-attention",
            "05-EDA-특성공학": "05-eda-feature-engineering",
            "06-모델-해석": "06-model-interpretation",
            "07-강화학습": "07-reinforcement-learning",
            "08-AI-윤리와-공정성": "08-ai-ethics-fairness",
        },
    ),
)


@dataclass(frozen=True)
class AppendixConfig:
    """부록 원본 → src/appendix 대응.

    slugs 의 순서가 책 목차(SUMMARY.md '# 부록' 절) 순서가 됩니다.
    README.md 는 index_slug 이름의 안내 페이지로 함께 들어갑니다.
    'a' 가 붙은 심화 글(01a, 02a …)은 바로 앞 글 아래에 들여써서 나열합니다.
    """

    source_dir: Path
    target_dir: Path
    slugs: dict[str, str]
    index_slug: str = "README"
    index_title: str | None = None


APPENDICES: tuple[AppendixConfig, ...] = (
    AppendixConfig(
        source_dir=ROOT / "부록",
        target_dir=SRC_ROOT / "appendix",
        slugs={
            "신경망-가족-한눈에": "neural-networks-overview",
            "정규화-총정리": "regularization-summary",
            "용어-사전": "glossary",
            "자주-묻는-질문": "faq",
            "더-나아가기": "further-reading",
        },
        index_title="부록 안내",
    ),
    AppendixConfig(
        source_dir=ROOT / "부록" / "수학",
        target_dir=SRC_ROOT / "appendix" / "math",
        index_title="수학 부록 안내",
        slugs={
            "00-수학이-왜-필요한가": "00-why-math",
            "01-중학수학-복습": "01-middle-school-math",
            "01a-중학수학-심화": "01a-middle-school-advanced",
            "02-고등수학-핵심": "02-high-school-math",
            "02a-고등수학-심화": "02a-high-school-advanced",
            "03-미분-입문": "03-calculus-intro",
            "03a-적분과-다변수": "03a-calculus-advanced",
            "04-선형대수-입문": "04-linear-algebra-intro",
            "04a-선형대수-심화": "04a-linear-algebra-advanced",
            "05-확률통계-입문": "05-probability-stats-intro",
            "05a-확률통계-심화": "05a-probability-stats-advanced",
            "06-ML에-적용하기": "06-ml-application",
            "07-최적화-심화": "07-optimization-advanced",
        },
    ),
)


LINK_PATTERN = re.compile(r"(!?\[[^\]]+\])\(([^)]+)\)")


def write_text_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def extract_markdown_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def extract_python_title(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
        docstring = ast.get_docstring(module)
    except SyntaxError:
        docstring = None

    if docstring:
        first_line = docstring.strip().splitlines()[0].strip()
        if first_line:
            return first_line

    return path.stem.replace("_", " ")


def collect_practice_sources(practice_dir: Path) -> tuple[Path, ...]:
    allowed_names = {"Dockerfile", "requirements.txt"}
    return tuple(
        sorted(
            path
            for path in practice_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.name != "README.md"
            and path.suffix != ".pyc"
            and (path.suffix == ".py" or path.name in allowed_names)
        )
    )


def practice_target_name(source_path: Path) -> str:
    if source_path.suffix == ".py":
        return f"{source_path.stem}.md"
    if source_path.name == "Dockerfile":
        return "Dockerfile.md"
    return f"{source_path.name.replace('.', '_')}.md"


def practice_target_path(chapter: ChapterConfig, source_path: Path) -> Path:
    return chapter.target_dir / "practice" / practice_target_name(source_path)


def extract_practice_title(source_path: Path) -> str:
    if source_path.suffix == ".py":
        return extract_python_title(source_path)
    if source_path.name == "Dockerfile":
        return "Docker 이미지 빌드 파일 (Dockerfile)"
    if source_path.name == "requirements.txt":
        return "실습 의존성 목록 (requirements.txt)"
    return f"실습 자료: {source_path.name}"


def practice_code_fence(source_path: Path) -> str:
    if source_path.suffix == ".py":
        return "python"
    if source_path.name == "Dockerfile":
        return "dockerfile"
    if source_path.suffix in {".yml", ".yaml"}:
        return "yaml"
    if source_path.suffix == ".json":
        return "json"
    return "text"


def relative_key(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_theory_pairs(theory_dir: Path, theory_glob: str, slugs: dict[str, str], target_dir: Path) -> tuple[tuple[Path, Path], ...]:
    """원본 이론 파일마다 slugs 로 책용 파일 경로를 정합니다. 빠진 항목은 바로 알려 줍니다."""
    sources = sorted(path for path in theory_dir.glob(theory_glob) if path.name != "README.md")
    missing = [path.stem for path in sources if path.stem not in slugs]
    if missing:
        raise RuntimeError(
            f"{theory_dir.relative_to(ROOT)} 에 책용 파일 이름(slug)이 정해지지 않은 파일이 있습니다: {missing}\n"
            f"scripts/sync_book.py 의 해당 장 slugs 에 '원본이름': 'english-slug' 를 추가해 주세요."
        )
    unused = sorted(set(slugs) - {path.stem for path in sources})
    if unused:
        raise RuntimeError(
            f"{theory_dir.relative_to(ROOT)} 에 없는 파일이 slugs 에 남아 있습니다: {unused}\n"
            f"scripts/sync_book.py 의 slugs 에서 지우거나 파일 이름을 확인해 주세요."
        )
    order = {stem: index for index, stem in enumerate(slugs)}
    sources.sort(key=lambda path: order[path.stem])  # slugs 에 적은 순서 = 책 목차 순서
    return tuple((path, target_dir / f"{slugs[path.stem]}.md") for path in sources)


def resolve_appendix_pairs(appendix: AppendixConfig) -> tuple[tuple[Path, Path], ...]:
    """부록 폴더의 README.md(안내 페이지)와 본문 글들을 책용 경로에 대응시킵니다."""
    pairs: list[tuple[Path, Path]] = []
    index_source = appendix.source_dir / "README.md"
    if index_source.exists():
        pairs.append((index_source, appendix.target_dir / f"{appendix.index_slug}.md"))
    pairs.extend(resolve_theory_pairs(appendix.source_dir, "*.md", appendix.slugs, appendix.target_dir))
    return tuple(pairs)


def build_link_maps() -> tuple[dict[str, Path], dict[Path, tuple[tuple[Path, Path], ...]], dict[Path, tuple[Path, ...]], dict[Path, Path]]:
    file_map: dict[str, Path] = {}
    chapter_theory_pairs: dict[Path, tuple[tuple[Path, Path], ...]] = {}
    chapter_practice_sources: dict[Path, tuple[Path, ...]] = {}
    chapter_readmes: dict[Path, Path] = {}

    for chapter in CHAPTERS:
        pairs = resolve_theory_pairs(chapter.theory_dir, chapter.theory_glob, chapter.slugs, chapter.target_dir)
        chapter_theory_pairs[chapter.source_dir] = pairs
        chapter_readmes[chapter.source_dir] = chapter.target_dir / "README.md"
        file_map[relative_key(chapter.source_dir)] = chapter.target_dir / "README.md"
        file_map[relative_key(chapter.source_dir / "README.md")] = chapter.target_dir / "README.md"
        if chapter.theory_dir != chapter.source_dir:
            file_map[relative_key(chapter.theory_dir)] = chapter.target_dir / "README.md"

        for source_path, target_path in pairs:
            file_map[relative_key(source_path)] = target_path

        practice_sources: tuple[Path, ...] = ()
        if chapter.practice_dir is not None:
            practice_sources = collect_practice_sources(chapter.practice_dir)
            file_map[relative_key(chapter.practice_dir)] = chapter.target_dir / "practice" / "README.md"
            for source_path in practice_sources:
                file_map[relative_key(source_path)] = practice_target_path(chapter, source_path)

        chapter_practice_sources[chapter.source_dir] = practice_sources

    for appendix in APPENDICES:
        for source_path, target_path in resolve_appendix_pairs(appendix):
            file_map[relative_key(source_path)] = target_path
        file_map.setdefault(relative_key(appendix.source_dir), appendix.target_dir / f"{appendix.index_slug}.md")
        file_map.setdefault(relative_key(appendix.source_dir / "README.md"), appendix.target_dir / f"{appendix.index_slug}.md")

    file_map[relative_key(ROOT / "README.md")] = SRC_ROOT / "intro.md"
    file_map[relative_key(ROOT / "CURRICULUM.md")] = SRC_ROOT / "intro.md"

    return file_map, chapter_theory_pairs, chapter_practice_sources, chapter_readmes


def rewrite_markdown_links(content: str, source_path: Path, target_path: Path, file_map: dict[str, Path]) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        raw_target = match.group(2).strip()

        if raw_target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        target_only, anchor = (raw_target.split("#", 1) + [""])[:2]
        resolved = (source_path.parent / target_only).resolve()
        try:
            resolved_key = relative_key(resolved)
        except ValueError:
            return match.group(0)

        mapped_target = file_map.get(resolved_key)
        if mapped_target is None and resolved.is_dir():
            mapped_target = file_map.get(f"{resolved_key}/README.md")

        if mapped_target is None:
            return match.group(0)

        rewritten = os.path.relpath(mapped_target, start=target_path.parent).replace(os.sep, "/")
        if anchor:
            rewritten = f"{rewritten}#{anchor}"
        return f"{label}({rewritten})"

    return LINK_PATTERN.sub(replace, content)


def sync_readme(chapter: ChapterConfig, link_map: dict[str, Path], changed: list[Path]) -> None:
    source_path = chapter.source_dir / "README.md"
    if not source_path.exists():
        return
    target_path = chapter.target_dir / "README.md"
    content = rewrite_markdown_links(source_path.read_text(encoding="utf-8"), source_path, target_path, link_map)
    if write_text_if_changed(target_path, content):
        changed.append(target_path)


def sync_theory(chapter: ChapterConfig, pairs: tuple[tuple[Path, Path], ...], link_map: dict[str, Path], changed: list[Path]) -> list[tuple[str, Path]]:
    synced: list[tuple[str, Path]] = []
    expected = {target_path for _, target_path in pairs} | {chapter.target_dir / "README.md"}
    for source_path, target_path in pairs:
        content = rewrite_markdown_links(source_path.read_text(encoding="utf-8"), source_path, target_path, link_map)
        if write_text_if_changed(target_path, content):
            changed.append(target_path)
        synced.append((extract_markdown_title(source_path), target_path))

    # 원본에서 사라진 이론 파일의 책용 사본은 지워서 고아 페이지를 막습니다.
    if chapter.target_dir.exists():
        for stale in sorted(chapter.target_dir.glob("*.md")):
            if stale not in expected:
                stale.unlink()
                changed.append(stale)
    return synced


def sync_appendix(link_map: dict[str, Path], changed: list[Path]) -> list[tuple[AppendixConfig, list[tuple[str, Path]]]]:
    entries: list[tuple[AppendixConfig, list[tuple[str, Path]]]] = []
    for appendix in APPENDICES:
        pairs = resolve_appendix_pairs(appendix)
        expected = {target_path for _, target_path in pairs}
        synced: list[tuple[str, Path]] = []
        for source_path, target_path in pairs:
            content = rewrite_markdown_links(source_path.read_text(encoding="utf-8"), source_path, target_path, link_map)
            if write_text_if_changed(target_path, content):
                changed.append(target_path)
            title = appendix.index_title if (source_path.name == "README.md" and appendix.index_title) else extract_markdown_title(source_path)
            synced.append((title, target_path))
        if appendix.target_dir.exists():
            for stale in sorted(appendix.target_dir.glob("*.md")):
                if stale not in expected:
                    stale.unlink()
                    changed.append(stale)
        entries.append((appendix, synced))
    return entries


def render_appendix_summary(appendix_entries: list[tuple[AppendixConfig, list[tuple[str, Path]]]]) -> str:
    """'# 부록' 절을 만듭니다. 심화 글(파일 이름에 'a' 가 붙은 것)은 앞 글 아래에 들여씁니다."""
    blocks: list[str] = []
    for appendix, synced in appendix_entries:
        lines: list[str] = []
        for title, target_path in synced:
            rel_path = target_path.relative_to(SRC_ROOT).as_posix()
            stem = target_path.stem
            advanced = re.match(r"^\d+a-", stem) is not None
            indent = "  " if advanced else ""
            lines.append(f"{indent}- [{title}](./{rel_path})")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def render_practice_page(source_path: Path) -> str:
    title = extract_practice_title(source_path)
    source_rel = source_path.relative_to(ROOT).as_posix()
    code = source_path.read_text(encoding="utf-8").rstrip()
    fence = practice_code_fence(source_path)
    return (
        f"# {title}\n\n"
        f"- 원본 파일: `{source_rel}`\n"
        f"- 동기화 방식: 강의 원본에서 자동 생성\n\n"
        "아래는 생략 없이 전체 코드입니다.\n\n"
        f"```{fence}\n"
        f"{code}\n"
        "```\n"
    )


def render_practice_index(chapter: ChapterConfig, entries: list[tuple[str, Path]]) -> str:
    lines = [
        "# 실습 파일 안내",
        "",
        f"- 동기화 원본: `{chapter.practice_dir.relative_to(ROOT).as_posix()}/`",
        "- 동기화 방식: 강의 원본에서 자동 생성",
        "- 아래 페이지들은 모두 생략 없는 전체 코드를 포함합니다.",
        "",
    ]

    practice_root = chapter.target_dir / "practice"
    for title, target_path in entries:
        rel_path = target_path.relative_to(practice_root).as_posix()
        lines.append(f"- [{title}](./{rel_path})")

    lines.append("")
    return "\n".join(lines)


def sync_practice(chapter: ChapterConfig, practice_sources: tuple[Path, ...], changed: list[Path]) -> list[tuple[str, Path]]:
    if not practice_sources:
        return []

    practice_root = chapter.target_dir / "practice"
    previous: dict[Path, str] = {}
    if practice_root.exists():
        previous = {path: path.read_text(encoding="utf-8") for path in practice_root.glob("*.md")}
        shutil.rmtree(practice_root)
    practice_root.mkdir(parents=True, exist_ok=True)

    synced: list[tuple[str, Path]] = []
    for source_path in practice_sources:
        target_path = practice_target_path(chapter, source_path)
        content = render_practice_page(source_path)
        if content != previous.get(target_path):
            changed.append(target_path)
        write_text_if_changed(target_path, content)
        synced.append((extract_practice_title(source_path), target_path))

    index_path = practice_root / "README.md"
    index_content = render_practice_index(chapter, synced)
    if index_content != previous.get(index_path):
        changed.append(index_path)
    write_text_if_changed(index_path, index_content)
    return synced


def build_summary(chapter_entries: list[tuple[ChapterConfig, Path | None, list[tuple[str, Path]], list[tuple[str, Path]]]], appendix_section: str) -> str:
    lines = [
        "# 인공지능 기초 (Fundamentals of Artificial Intelligence)",
        "",
        "[서문](./intro.md)",
        "",
    ]

    for chapter, readme_target, theory_entries, practice_entries in chapter_entries:
        lines.append(chapter.summary_heading)
        lines.append("")

        if readme_target is not None and readme_target.exists():
            readme_rel = readme_target.relative_to(SRC_ROOT).as_posix()
            lines.append(f"- [이 장 안내](./{readme_rel})")

        for title, target_path in theory_entries:
            rel_path = target_path.relative_to(SRC_ROOT).as_posix()
            lines.append(f"- [{title}](./{rel_path})")

        for title, target_path in practice_entries:
            rel_path = target_path.relative_to(SRC_ROOT).as_posix()
            lines.append(f"- [실습: {title}](./{rel_path})")

        lines.append("")

    lines.append("# 부록")
    lines.append("")
    lines.append(appendix_section)
    lines.append("")
    return "\n".join(lines)


def sync_book_sources(verbose: bool = False) -> list[Path]:
    changed: list[Path] = []
    link_map, chapter_theory_pairs, chapter_practice_sources, chapter_readmes = build_link_maps()
    chapter_entries: list[tuple[ChapterConfig, Path | None, list[tuple[str, Path]], list[tuple[str, Path]]]] = []

    for chapter in CHAPTERS:
        sync_readme(chapter, link_map, changed)
        theory_entries = sync_theory(chapter, chapter_theory_pairs[chapter.source_dir], link_map, changed)
        practice_entries = sync_practice(chapter, chapter_practice_sources[chapter.source_dir], changed)
        readme_target = chapter_readmes.get(chapter.source_dir)
        chapter_entries.append((chapter, readme_target, theory_entries, practice_entries))

    appendix_entries = sync_appendix(link_map, changed)

    summary_content = build_summary(chapter_entries, render_appendix_summary(appendix_entries))
    if write_text_if_changed(SUMMARY_PATH, summary_content):
        changed.append(SUMMARY_PATH)

    if verbose:
        print(f"synced {len(changed)} generated files")
        for path in changed:
            print(path.relative_to(ROOT).as_posix())

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync lecture sources into the mdBook src tree.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-file output.")
    args = parser.parse_args()

    changed = sync_book_sources(verbose=not args.quiet)
    print(f"book sync complete: {len(changed)} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
