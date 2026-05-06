from __future__ import annotations

import subprocess
import time
from pathlib import Path

from sync_book import CHAPTERS, ROOT, collect_practice_sources, sync_book_sources


def iter_source_files() -> list[Path]:
    files: list[Path] = [ROOT / "src" / "intro.md", ROOT / "book.toml"]
    for chapter in CHAPTERS:
        readme_path = chapter.source_dir / "README.md"
        if readme_path.exists():
            files.append(readme_path)
        files.extend(sorted(chapter.theory_dir.glob(chapter.theory_glob)))
        if chapter.practice_dir is not None:
            files.extend(collect_practice_sources(chapter.practice_dir))
    return files


def snapshot() -> dict[str, int]:
    return {
        path.relative_to(ROOT).as_posix(): path.stat().st_mtime_ns
        for path in iter_source_files()
        if path.exists()
    }


def main() -> int:
    sync_book_sources(verbose=True)
    server = subprocess.Popen(["mdbook", "serve"], cwd=ROOT)
    previous = snapshot()

    try:
        while True:
            time.sleep(1.0)
            current = snapshot()
            if current != previous:
                print("source change detected: syncing book sources")
                sync_book_sources(verbose=True)
                previous = snapshot()
    except KeyboardInterrupt:
        print("stopping watcher")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())