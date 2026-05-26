from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote


FENCE_RE = re.compile(r"^(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXCLUDED_PARTS = {".git", ".venv", "node_modules", "target", "dist"}
IGNORED_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:")


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_markdown_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            for child in sorted(path.rglob("*.md")):
                if is_excluded(child):
                    continue
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(child)
        elif path.is_file() and path.suffix == ".md":
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


def strip_code_fences(lines: list[str]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(lines, start=1):
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            result.append((line_number, line))
    return result


def slugify(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
    heading = re.sub(r"[`*_~]", "", heading)
    heading = re.sub(r"[^a-z0-9\s-]", "", heading)
    heading = re.sub(r"\s", "-", heading)
    return heading.strip("-")


def heading_anchors(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]

    anchors: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    lines = path.read_text(encoding="utf-8").splitlines()
    for _, line in strip_code_fences(lines):
        match = HEADING_RE.match(line)
        if not match:
            continue
        slug = slugify(match.group(2))
        if not slug:
            continue
        count = counts[slug]
        counts[slug] += 1
        if count:
            slug = f"{slug}-{count}"
        anchors.add(slug)

    cache[path] = anchors
    return anchors


def normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if " \"" in target:
        target = target.split(" \"", 1)[0]
    elif " '" in target:
        target = target.split(" '", 1)[0]
    return unquote(target)


def validate_link(
    source: Path,
    line_number: int,
    target: str,
    anchor_cache: dict[Path, set[str]],
) -> list[str]:
    errors: list[str] = []
    target = normalize_target(target)

    if not target or target.startswith(IGNORED_SCHEMES):
        return errors
    if target.startswith("javascript:"):
        return [f"{source}:{line_number}: disallowed link target '{target}'"]

    if target.startswith("#"):
        anchor = target[1:]
        if anchor not in heading_anchors(source, anchor_cache):
            errors.append(
                f"{source}:{line_number}: missing anchor '#{anchor}' in {source}"
            )
        return errors

    path_part, _, anchor = target.partition("#")
    resolved = (source.parent / path_part).resolve()
    if not resolved.exists():
        errors.append(f"{source}:{line_number}: missing target '{target}'")
        return errors

    if anchor and resolved.suffix == ".md":
        if anchor not in heading_anchors(resolved, anchor_cache):
            errors.append(
                f"{source}:{line_number}: missing anchor '#{anchor}' in {resolved}"
            )

    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_markdown_links.py <file-or-dir> [...]")
        return 2

    markdown_files = iter_markdown_files(argv)
    anchor_cache: dict[Path, set[str]] = {}
    errors: list[str] = []

    for path in markdown_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in strip_code_fences(lines):
            for match in LINK_RE.finditer(line):
                errors.extend(validate_link(path, line_number, match.group(1), anchor_cache))

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"checked {len(markdown_files)} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))