from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
SUMMARY_PATH = SRC_ROOT / "SUMMARY.md"


@dataclass(frozen=True)
class ChapterConfig:
    summary_heading: str
    source_dir: Path
    theory_dir: Path
    target_dir: Path
    theory_glob: str
    practice_dir: Path | None = None


CHAPTERS: tuple[ChapterConfig, ...] = (
    ChapterConfig(
        summary_heading="# 시작하기 전에",
        source_dir=ROOT / "00-시작하기-전에",
        theory_dir=ROOT / "00-시작하기-전에",
        target_dir=SRC_ROOT / "00-intro",
        theory_glob="[0-9][0-9]-*.md",
    ),
    ChapterConfig(
        summary_heading="# 1장. 파이썬 기초",
        source_dir=ROOT / "01-파이썬-기초",
        theory_dir=ROOT / "01-파이썬-기초" / "이론",
        target_dir=SRC_ROOT / "01-python-basics",
        theory_glob="*.md",
        practice_dir=ROOT / "01-파이썬-기초" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 2장. 머신러닝 — 분류",
        source_dir=ROOT / "02-머신러닝-분류",
        theory_dir=ROOT / "02-머신러닝-분류" / "이론",
        target_dir=SRC_ROOT / "02-ml-classification",
        theory_glob="*.md",
        practice_dir=ROOT / "02-머신러닝-분류" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 3장. 머신러닝 — 회귀",
        source_dir=ROOT / "03-머신러닝-회귀",
        theory_dir=ROOT / "03-머신러닝-회귀" / "이론",
        target_dir=SRC_ROOT / "03-ml-regression",
        theory_glob="*.md",
        practice_dir=ROOT / "03-머신러닝-회귀" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 4장. 딥러닝 — 분류",
        source_dir=ROOT / "04-딥러닝-분류",
        theory_dir=ROOT / "04-딥러닝-분류" / "이론",
        target_dir=SRC_ROOT / "04-dl-classification",
        theory_glob="*.md",
        practice_dir=ROOT / "04-딥러닝-분류" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 5장. 딥러닝 — 회귀",
        source_dir=ROOT / "05-딥러닝-회귀",
        theory_dir=ROOT / "05-딥러닝-회귀" / "이론",
        target_dir=SRC_ROOT / "05-dl-regression",
        theory_glob="*.md",
        practice_dir=ROOT / "05-딥러닝-회귀" / "실습",
    ),
    ChapterConfig(
        summary_heading="# 6장. 프로덕션으로 가는 길",
        source_dir=ROOT / "06-프로덕션으로-가는-길",
        theory_dir=ROOT / "06-프로덕션으로-가는-길" / "이론",
        target_dir=SRC_ROOT / "06-production",
        theory_glob="*.md",
        practice_dir=ROOT / "06-프로덕션으로-가는-길" / "실습",
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


def read_appendix_section() -> str:
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    marker = "# 부록"
    if marker not in summary_text:
        raise RuntimeError("src/SUMMARY.md 에서 '# 부록' 섹션을 찾지 못했습니다.")
    return summary_text.split(marker, maxsplit=1)[1].strip()


def collect_theory_targets(target_dir: Path) -> list[Path]:
    return sorted(path for path in target_dir.glob("*.md") if path.name != "README.md")


def relative_key(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_link_maps() -> tuple[dict[str, Path], dict[Path, tuple[Path, ...]], dict[Path, tuple[Path, ...]], dict[Path, Path]]:
    file_map: dict[str, Path] = {}
    chapter_theory_sources: dict[Path, tuple[Path, ...]] = {}
    chapter_practice_sources: dict[Path, tuple[Path, ...]] = {}
    chapter_readmes: dict[Path, Path] = {}

    for chapter in CHAPTERS:
        theory_sources = tuple(sorted(chapter.theory_dir.glob(chapter.theory_glob)))
        theory_targets = tuple(collect_theory_targets(chapter.target_dir))
        if len(theory_sources) != len(theory_targets):
            raise RuntimeError(
                f"{chapter.target_dir.relative_to(ROOT)} 파일 수가 맞지 않습니다: "
                f"source={len(theory_sources)} target={len(theory_targets)}"
            )

        chapter_theory_sources[chapter.source_dir] = theory_sources
        chapter_readmes[chapter.source_dir] = chapter.target_dir / "README.md"
        file_map[relative_key(chapter.source_dir)] = chapter.target_dir / "README.md"
        file_map[relative_key(chapter.source_dir / "README.md")] = chapter.target_dir / "README.md"

        for source_path, target_path in zip(theory_sources, theory_targets, strict=True):
            file_map[relative_key(source_path)] = target_path

        practice_sources: tuple[Path, ...] = ()
        if chapter.practice_dir is not None:
            practice_sources = collect_practice_sources(chapter.practice_dir)
            file_map[relative_key(chapter.practice_dir)] = chapter.target_dir / "practice" / "README.md"
            for source_path in practice_sources:
                file_map[relative_key(source_path)] = practice_target_path(chapter, source_path)

        chapter_practice_sources[chapter.source_dir] = practice_sources

    return file_map, chapter_theory_sources, chapter_practice_sources, chapter_readmes


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
            mapped_target = file_map.get(resolved_key)
            if mapped_target is None:
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


def sync_theory(chapter: ChapterConfig, theory_sources: tuple[Path, ...], link_map: dict[str, Path], changed: list[Path]) -> list[tuple[str, Path]]:
    theory_targets = collect_theory_targets(chapter.target_dir)
    synced: list[tuple[str, Path]] = []
    for source_path, target_path in zip(theory_sources, theory_targets, strict=True):
        content = rewrite_markdown_links(source_path.read_text(encoding="utf-8"), source_path, target_path, link_map)
        if write_text_if_changed(target_path, content):
            changed.append(target_path)
        synced.append((extract_markdown_title(source_path), target_path))
    return synced


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

    for title, target_path in entries:
        rel_path = target_path.relative_to(chapter.target_dir).as_posix()
        lines.append(f"- [{title}](./{rel_path})")

    lines.append("")
    return "\n".join(lines)


def sync_practice(chapter: ChapterConfig, practice_sources: tuple[Path, ...], changed: list[Path]) -> list[tuple[str, Path]]:
    if not practice_sources:
        return []

    practice_root = chapter.target_dir / "practice"
    if practice_root.exists():
        shutil.rmtree(practice_root)
    practice_root.mkdir(parents=True, exist_ok=True)

    synced: list[tuple[str, Path]] = []
    for source_path in practice_sources:
        target_path = practice_target_path(chapter, source_path)
        content = render_practice_page(source_path)
        write_text_if_changed(target_path, content)
        changed.append(target_path)
        synced.append((extract_practice_title(source_path), target_path))

    index_path = practice_root / "README.md"
    index_content = render_practice_index(chapter, synced)
    write_text_if_changed(index_path, index_content)
    changed.append(index_path)
    return synced


def build_summary(chapter_entries: list[tuple[ChapterConfig, Path | None, list[tuple[str, Path]], list[tuple[str, Path]]]]) -> str:
    appendix_section = read_appendix_section()
    lines = [
        "# 인공지능 기초 (Fundamentals of Artificial Intelligence)",
        "",
        "[서문](./intro.md)",
        "",
    ]

    for chapter, readme_target, theory_entries, practice_entries in chapter_entries:
        lines.append(chapter.summary_heading)
        lines.append("")

        if readme_target is not None:
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
    link_map, chapter_theory_sources, chapter_practice_sources, chapter_readmes = build_link_maps()
    chapter_entries: list[tuple[ChapterConfig, Path | None, list[tuple[str, Path]], list[tuple[str, Path]]]] = []

    for chapter in CHAPTERS:
        sync_readme(chapter, link_map, changed)
        theory_entries = sync_theory(chapter, chapter_theory_sources[chapter.source_dir], link_map, changed)
        practice_entries = sync_practice(chapter, chapter_practice_sources[chapter.source_dir], changed)
        readme_target = chapter_readmes.get(chapter.source_dir)
        chapter_entries.append((chapter, readme_target, theory_entries, practice_entries))

    summary_content = build_summary(chapter_entries)
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