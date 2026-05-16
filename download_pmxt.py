#!/usr/bin/env python3
"""
Download Polymarket orderbook data from the pmxt archive.
Stores files to ~/data/pmxt/polymarket, skipping files already present.
"""

import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from html.parser import HTMLParser


ARCHIVE_BASE = "https://archive.pmxt.dev/Polymarket"
DEST_DIR = Path.home() / "data" / "pmxt" / "polymarket"


class FileListParser(HTMLParser):
    """Extract (filename, url) pairs and the total page count from an archive index page."""

    def __init__(self):
        super().__init__()
        self.files: list[tuple[str, str]] = []  # (filename, url)
        self.total_pages: int = 1
        self._in_a = False
        self._current_href = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href.startswith("https://r2.pmxt.dev/"):
                self._in_a = True
                self._current_href = href

    def handle_data(self, data):
        if self._in_a:
            filename = data.strip()
            if filename.endswith(".parquet"):
                self.files.append((filename, self._current_href))

    def handle_endtag(self, tag):
        if tag == "a":
            self._in_a = False
            self._current_href = ""

    def feed(self, data: str):
        super().feed(data)
        # Extract total pages from "Page X of Y"
        match = re.search(r"Page\s+\d+\s+of\s+(\d+)", data)
        if match:
            self.total_pages = int(match.group(1))


def fetch_page(page: int) -> str:
    url = ARCHIVE_BASE if page == 1 else f"{ARCHIVE_BASE}?page={page}"
    req = urllib.request.Request(url, headers={"User-Agent": "pmxt-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def collect_all_files() -> list[tuple[str, str]]:
    """Page through the archive and return every (filename, url) entry."""
    print("Fetching page 1 to determine total pages...")
    html = fetch_page(1)
    parser = FileListParser()
    parser.feed(html)
    total_pages = parser.total_pages
    all_files = list(parser.files)
    print(f"  Found {len(all_files)} files on page 1 (total pages: {total_pages})")

    for page in range(2, total_pages + 1):
        print(f"Fetching page {page}/{total_pages}...")
        try:
            html = fetch_page(page)
        except Exception as exc:
            print(f"  ERROR fetching page {page}: {exc}", file=sys.stderr)
            time.sleep(2)
            continue
        p = FileListParser()
        p.feed(html)
        all_files.extend(p.files)
        print(f"  Found {len(p.files)} files on page {page}")
        time.sleep(0.3)  # be polite

    return all_files


def download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pmxt-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            with open(tmp, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = downloaded / total * 100
                        mb_done = downloaded / 1e6
                        mb_total = total / 1e6
                        print(
                            f"\r  {mb_done:.1f}/{mb_total:.1f} MB  ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
        print()  # newline after progress
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {DEST_DIR}\n")

    all_files = collect_all_files()
    print(f"\nTotal files in archive: {len(all_files)}")

    to_download = [
        (name, url) for name, url in all_files if not (DEST_DIR / name).exists()
    ]
    already_have = len(all_files) - len(to_download)
    print(f"Already downloaded:     {already_have}")
    print(f"To download:            {len(to_download)}\n")

    if not to_download:
        print("Nothing to do.")
        return

    for i, (name, url) in enumerate(to_download, 1):
        dest = DEST_DIR / name
        print(f"[{i}/{len(to_download)}] {name}")
        try:
            download_file(url, dest)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
