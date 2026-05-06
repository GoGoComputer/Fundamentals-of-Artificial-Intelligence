from __future__ import annotations

import subprocess

from sync_book import ROOT, sync_book_sources


def main() -> int:
    sync_book_sources(verbose=True)
    completed = subprocess.run(["mdbook", "build"], cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())